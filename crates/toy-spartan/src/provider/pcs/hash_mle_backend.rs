//! Backend interface for Hash-MLE PCS: field bridge + digest.
//! Two implementations live in this crate:
//!  - ff_keccak: uses E::Scalar directly + Keccak256
//!  - p3_poseidon2_goldilocks: converts E::Scalar <-> p3_goldilocks::Goldilocks and hashes with Poseidon2

use crate::traits::{Engine, transcript::TranscriptReprTrait};
use ff::{Field, PrimeField};
use serde::{Deserialize, Serialize};

#[cfg(feature = "p3_backend")]
use p3_goldilocks::Goldilocks as GF;

#[cfg(feature = "p3_backend")]
use p3_field::{PrimeCharacteristicRing, PrimeField64, integers::QuotientMap};

#[cfg(feature = "p3_backend")]
use p3_goldilocks::Poseidon2Goldilocks;

#[cfg(feature = "p3_backend")]
use p3_symmetric::Permutation;

#[cfg(feature = "p3_backend")]
use once_cell::sync::Lazy;

#[cfg(feature = "p3_backend")]
use rand_chacha_p3::ChaCha8Rng;
#[cfg(feature = "p3_backend")]
use rand_chacha_p3::rand_core::SeedableRng;

/// Poseidon2 width (Goldilocks, 64-bit prime). Supported: {8,12,16}. We use 12 to fit domain+len+8 words.
#[cfg(feature = "p3_backend")]
const POSEIDON2_WIDTH: usize = 12;

/// Fixed seed to derive round constants deterministically (reproducible hash).
/// Change only if you intentionally rotate parameters.
#[cfg(feature = "p3_backend")]
const POSEIDON2_PARAM_SEED_V1: u64 = 0x5_7045_3F1A_CB7D_12;

#[cfg(feature = "p3_backend")]
static POSEIDON2_GOLDI_12: Lazy<Poseidon2Goldilocks<POSEIDON2_WIDTH>> = Lazy::new(|| {
  // Deterministic constants for stable hashing across runs.
  let mut rng = ChaCha8Rng::seed_from_u64(POSEIDON2_PARAM_SEED_V1);
  Poseidon2Goldilocks::<POSEIDON2_WIDTH>::new_from_rng_128(&mut rng)
});

/// Domain tags injected into state[0] to separate distinct uses.
#[cfg(feature = "p3_backend")]
const DOMAIN_LEAF: u64 = 0x_6C65_6166_2F6D_6C65; // ASCII-ish "leaf/mle" (little-endian as u64)
#[cfg(feature = "p3_backend")]
const DOMAIN_NODE: u64 = 0x_6E6F_6465_2F6D_6C65; // ASCII-ish "node/mle"

/// 32-byte digest used by the Merkle tree / commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Digest32(pub [u8; 32]);

impl<G: crate::traits::Group> TranscriptReprTrait<G> for Digest32 {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.0.to_vec()
  }
}

/// Backends are parameterized by the engine E, so they can see E::Scalar at the boundary.
pub trait MleBackend<E: Engine> {
  /// Internal field element type used by this backend.
  type FE: Copy + Send + Sync + 'static;

  /// Convert from the engine's scalar (ff) to the backend field.
  fn fe_from_ff(x: &E::Scalar) -> Self::FE;

  /// Convert from backend field back to the engine's scalar (ff).
  fn fe_to_ff(x: &Self::FE) -> E::Scalar;

  /// Field ops
  fn zero() -> Self::FE;
  fn one() -> Self::FE;
  fn add(a: Self::FE, b: Self::FE) -> Self::FE;
  fn sub(a: Self::FE, b: Self::FE) -> Self::FE;
  fn mul(a: Self::FE, b: Self::FE) -> Self::FE;

  /// Hash a leaf field element into a 32-byte digest.
  fn leaf_hash(x: &Self::FE) -> Digest32;

  /// Hash/consolidate two child digests into a parent digest (2-to-1).
  fn node_hash(l: &Digest32, r: &Digest32) -> Digest32;
}

// -----------------------------
// ff + Keccak (existing behavior)
// -----------------------------
pub struct BackendFfKeccak<E: Engine>(core::marker::PhantomData<E>);

impl<E: Engine> MleBackend<E> for BackendFfKeccak<E> {
  type FE = E::Scalar;

  #[inline]
  fn fe_from_ff(x: &E::Scalar) -> Self::FE {
    *x
  }
  #[inline]
  fn fe_to_ff(x: &Self::FE) -> E::Scalar {
    *x
  }

  #[inline]
  fn zero() -> Self::FE {
    E::Scalar::ZERO
  }
  #[inline]
  fn one() -> Self::FE {
    E::Scalar::ONE
  }
  #[inline]
  fn add(a: Self::FE, b: Self::FE) -> Self::FE {
    a + b
  }
  #[inline]
  fn sub(a: Self::FE, b: Self::FE) -> Self::FE {
    a - b
  }
  #[inline]
  fn mul(a: Self::FE, b: Self::FE) -> Self::FE {
    a * b
  }

  fn leaf_hash(x: &Self::FE) -> Digest32 {
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(b"mle/leaf");
    // Avoid per-leaf heap allocations from `to_transcript_bytes()` (hot path).
    let repr = x.to_repr();
    hasher.update(repr.as_ref());
    Digest32(hasher.finalize().into())
  }

  fn node_hash(l: &Digest32, r: &Digest32) -> Digest32 {
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(b"mle/node");
    hasher.update(&l.0);
    hasher.update(&r.0);
    Digest32(hasher.finalize().into())
  }
}

// -----------------------------------------
// p3/Goldilocks + Poseidon2 (feature-gated)
// -----------------------------------------
#[cfg(feature = "p3_backend")]
pub struct BackendP3Poseidon2Goldi<E: Engine>(core::marker::PhantomData<E>);

// Type safety: P3 backend now has compile-time guarantee that E::Scalar = goldi::F

#[cfg(feature = "p3_backend")]
impl<E> MleBackend<E> for BackendP3Poseidon2Goldi<E>
where
  E: Engine<Scalar = crate::provider::goldi::F>,
{
  // Internal field is p3_goldilocks::Goldilocks
  type FE = p3_goldilocks::Goldilocks;

  #[inline]
  fn fe_from_ff(x: &E::Scalar) -> Self::FE {
    // Now we have compile-time guarantee that E::Scalar = goldi::F
    // Use total conversion via canonical u64 instead of truncating bytes
    GF::from_int(x.to_canonical_u64())
  }

  #[inline]
  fn fe_to_ff(x: &Self::FE) -> E::Scalar {
    E::Scalar::from(x.as_canonical_u64())
  }

  #[inline]
  fn zero() -> Self::FE {
    GF::ZERO
  }
  #[inline]
  fn one() -> Self::FE {
    GF::ONE
  }
  #[inline]
  fn add(a: Self::FE, b: Self::FE) -> Self::FE {
    a + b
  }
  #[inline]
  fn sub(a: Self::FE, b: Self::FE) -> Self::FE {
    a - b
  }
  #[inline]
  fn mul(a: Self::FE, b: Self::FE) -> Self::FE {
    a * b
  }

  fn leaf_hash(x: &Self::FE) -> Digest32 {
    poseidon2_hash_leaf(*x)
  }

  fn node_hash(l: &Digest32, r: &Digest32) -> Digest32 {
    // Interpret each 32-byte digest as four LE u64s → Goldilocks words
    let mut limbs: [GF; 8] = [GF::ZERO; 8];
    for (i, d) in [l, r].iter().enumerate() {
      for j in 0..4 {
        let off = 8 * j;
        let w = u64::from_le_bytes(d.0[off..off + 8].try_into().unwrap());
        limbs[i * 4 + j] = GF::from_int(w);
      }
    }
    poseidon2_hash_node(&limbs)
  }
}

/// Compress first 4 state words into 32 bytes (LE u64 each).
#[cfg(feature = "p3_backend")]
#[inline]
fn state4_to_digest32(state: &[GF; POSEIDON2_WIDTH]) -> Digest32 {
  let mut out = [0u8; 32];
  for i in 0..4 {
    out[i * 8..(i + 1) * 8].copy_from_slice(&state[i].as_canonical_u64().to_le_bytes());
  }
  Digest32(out)
}

/// Hash a single Goldilocks element as a Poseidon2 leaf.
/// State layout (width=12):
///   s[0] = DOMAIN_LEAF
///   s[1] = len = 1
///   s[2] = x
///   s[3..] = 0
#[cfg(feature = "p3_backend")]
#[inline]
fn poseidon2_hash_leaf(x: GF) -> Digest32 {
  let mut s = [GF::ZERO; POSEIDON2_WIDTH];
  s[0] = GF::from_int(DOMAIN_LEAF);
  s[1] = GF::from_int(1);
  s[2] = x;
  POSEIDON2_GOLDI_12.permute_mut(&mut s);
  state4_to_digest32(&s)
}

/// Hash two child digests into a parent digest.
/// Each child digest (32 bytes) is split into four LE u64s → Goldilocks words.
/// We expect exactly 8 words: [l0..l3, r0..r3].
/// State layout (width=12):
///   s[0] = DOMAIN_NODE
///   s[1] = len = 8
///   s[2..10] = words[0..8]
///   s[10..] = 0
#[cfg(feature = "p3_backend")]
#[inline]
fn poseidon2_hash_node(words: &[GF]) -> Digest32 {
  debug_assert_eq!(words.len(), 8, "node hash expects 8 Goldilocks words");
  let mut s = [GF::ZERO; POSEIDON2_WIDTH];
  s[0] = GF::from_int(DOMAIN_NODE);
  s[1] = GF::from_int(8);
  for i in 0..8 {
    s[2 + i] = words[i];
  }
  POSEIDON2_GOLDI_12.permute_mut(&mut s);
  state4_to_digest32(&s)
}

#[cfg(all(test, feature = "p3_backend"))]
mod poseidon2_sanity_tests {
  use super::*;

  #[test]
  fn poseidon2_leaf_is_deterministic() {
    let x = GF::from_int(123456789);
    let d1 = poseidon2_hash_leaf(x);
    let d2 = poseidon2_hash_leaf(x);
    assert_eq!(d1, d2);
  }

  #[test]
  fn poseidon2_node_is_deterministic() {
    let mut w = [GF::ZERO; 8];
    for i in 0..8 {
      w[i] = GF::from_int(i as u64 + 1);
    }
    let d1 = poseidon2_hash_node(&w);
    let d2 = poseidon2_hash_node(&w);
    assert_eq!(d1, d2);
  }
}
