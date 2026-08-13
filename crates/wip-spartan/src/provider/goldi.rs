//! This module implements a minimal Goldilocks field and dummy group for Hash-MLE PCS.
//! This provides an ff-compatible Goldilocks field and a unit point group (no MSMs needed).

use crate::traits::{Group, PrimeFieldExt, transcript::TranscriptReprTrait};
use ff::{Field, PrimeField, PrimeFieldBits};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::iter::{Product, Sum};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

/// Goldilocks prime: 2^64 - 2^32 + 1
const GOLDILOCKS_MODULUS: u64 = 0xFFFFFFFF00000001;

/// A Goldilocks field element (ff-compatible)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct F(u64);

impl F {
  /// Create a new field element from a u64, reducing modulo the Goldilocks prime
  pub fn new(val: u64) -> Self {
    Self(val % GOLDILOCKS_MODULUS)
  }

  /// Get the canonical u64 representation
  pub fn to_canonical_u64(self) -> u64 {
    self.0
  }

  /// Convert from canonical u64 (assumes input is already reduced)
  pub fn from_canonical_u64(val: u64) -> Self {
    debug_assert!(val < GOLDILOCKS_MODULUS);
    Self(val)
  }
}

impl Field for F {
  const ZERO: Self = Self(0);
  const ONE: Self = Self(1);

  fn random(mut rng: impl rand_core::RngCore) -> Self {
    // Rejection sampling (unbiased)
    loop {
      let x = rng.next_u64();
      if x < GOLDILOCKS_MODULUS {
        return Self::from_canonical_u64(x);
      }
    }
  }

  fn square(&self) -> Self {
    *self * *self
  }

  fn double(&self) -> Self {
    *self + *self
  }

  fn invert(&self) -> CtOption<Self> {
    if self.0 == 0 {
      CtOption::new(Self::ZERO, Choice::from(0))
    } else {
      // Use Fermat's little theorem: a^(p-1) = 1, so a^(-1) = a^(p-2)
      let inv = self.pow_vartime(&[GOLDILOCKS_MODULUS - 2]);
      CtOption::new(inv, Choice::from(1))
    }
  }

  fn sqrt(&self) -> CtOption<Self> {
    // For now, just return None (not needed for our use case)
    CtOption::new(Self::ZERO, Choice::from(0))
  }

  fn sqrt_ratio(_num: &Self, _div: &Self) -> (Choice, Self) {
    (Choice::from(0), Self::ZERO)
  }
}

impl std::ops::Add for F {
  type Output = Self;

  fn add(self, rhs: Self) -> Self {
    let sum = (self.0 as u128) + (rhs.0 as u128);
    if sum >= GOLDILOCKS_MODULUS as u128 {
      Self((sum - GOLDILOCKS_MODULUS as u128) as u64)
    } else {
      Self(sum as u64)
    }
  }
}

impl std::ops::Sub for F {
  type Output = Self;

  fn sub(self, rhs: Self) -> Self {
    if self.0 >= rhs.0 {
      Self(self.0 - rhs.0)
    } else {
      Self(self.0.wrapping_add(GOLDILOCKS_MODULUS).wrapping_sub(rhs.0))
    }
  }
}

impl std::ops::Mul for F {
  type Output = Self;

  fn mul(self, rhs: Self) -> Self {
    let prod = (self.0 as u128) * (rhs.0 as u128);
    Self((prod % GOLDILOCKS_MODULUS as u128) as u64)
  }
}

impl std::ops::Neg for F {
  type Output = Self;

  fn neg(self) -> Self {
    if self.0 == 0 {
      Self::ZERO
    } else {
      Self(GOLDILOCKS_MODULUS - self.0)
    }
  }
}

impl std::ops::AddAssign for F {
  fn add_assign(&mut self, rhs: Self) {
    *self = *self + rhs;
  }
}

impl std::ops::SubAssign for F {
  fn sub_assign(&mut self, rhs: Self) {
    *self = *self - rhs;
  }
}

impl std::ops::MulAssign for F {
  fn mul_assign(&mut self, rhs: Self) {
    *self = *self * rhs;
  }
}

// Reference operations for Field trait
impl std::ops::Add<&F> for F {
  type Output = F;
  fn add(self, rhs: &F) -> F {
    self + *rhs
  }
}

impl std::ops::Sub<&F> for F {
  type Output = F;
  fn sub(self, rhs: &F) -> F {
    self - *rhs
  }
}

impl std::ops::Mul<&F> for F {
  type Output = F;
  fn mul(self, rhs: &F) -> F {
    self * *rhs
  }
}

impl std::ops::AddAssign<&F> for F {
  fn add_assign(&mut self, rhs: &F) {
    *self = *self + *rhs;
  }
}

impl std::ops::SubAssign<&F> for F {
  fn sub_assign(&mut self, rhs: &F) {
    *self = *self - *rhs;
  }
}

impl std::ops::MulAssign<&F> for F {
  fn mul_assign(&mut self, rhs: &F) {
    *self = *self * *rhs;
  }
}

// Sum and Product for Field trait
impl Sum for F {
  fn sum<I: Iterator<Item = F>>(iter: I) -> F {
    iter.fold(F::ZERO, |acc, x| acc + x)
  }
}

impl<'a> Sum<&'a F> for F {
  fn sum<I: Iterator<Item = &'a F>>(iter: I) -> F {
    iter.fold(F::ZERO, |acc, x| acc + *x)
  }
}

impl Product for F {
  fn product<I: Iterator<Item = F>>(iter: I) -> F {
    iter.fold(F::ONE, |acc, x| acc * x)
  }
}

impl<'a> Product<&'a F> for F {
  fn product<I: Iterator<Item = &'a F>>(iter: I) -> F {
    iter.fold(F::ONE, |acc, x| acc * *x)
  }
}

impl ConditionallySelectable for F {
  fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
    Self(u64::conditional_select(&a.0, &b.0, choice))
  }
}

impl ConstantTimeEq for F {
  fn ct_eq(&self, other: &Self) -> Choice {
    self.0.ct_eq(&other.0)
  }
}

impl From<u64> for F {
  fn from(val: u64) -> Self {
    Self::new(val)
  }
}

impl PrimeField for F {
  type Repr = [u8; 8];

  const MODULUS: &'static str = "18446744069414584321"; // GOLDILOCKS_MODULUS in decimal
  const NUM_BITS: u32 = 64;
  const CAPACITY: u32 = 63;
  const TWO_INV: Self = Self(9223372034707292161); // (GOLDILOCKS_MODULUS + 1) / 2
  const MULTIPLICATIVE_GENERATOR: Self = Self(7);
  const S: u32 = 32;
  const ROOT_OF_UNITY: Self = Self(1753635133440165772);
  const ROOT_OF_UNITY_INV: Self = Self(8595233332735842049);
  const DELTA: Self = Self(1753635133440165772);

  fn from_repr(repr: Self::Repr) -> CtOption<Self> {
    let val = u64::from_le_bytes(repr);
    if val < GOLDILOCKS_MODULUS {
      CtOption::new(Self(val), Choice::from(1))
    } else {
      CtOption::new(Self::ZERO, Choice::from(0))
    }
  }

  fn to_repr(&self) -> Self::Repr {
    self.0.to_le_bytes()
  }

  fn is_odd(&self) -> Choice {
    Choice::from((self.0 & 1) as u8)
  }
}

impl PrimeFieldBits for F {
  type ReprBits = [u8; 8];

  fn to_le_bits(&self) -> ff::FieldBits<Self::ReprBits> {
    ff::FieldBits::new(self.to_repr())
  }

  fn char_le_bits() -> ff::FieldBits<Self::ReprBits> {
    ff::FieldBits::new(GOLDILOCKS_MODULUS.to_le_bytes())
  }
}

impl PrimeFieldExt for F {
  fn from_uniform(digest: &[u8]) -> Self {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8.min(digest.len())]);
    let val = u64::from_le_bytes(bytes);
    Self::new(val)
  }
}

/// A dummy point type that represents the identity element (no actual group operations)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitPoint;

impl Group for UnitPoint {
  type Base = F;
  type Scalar = F;

  fn group_params() -> (Self::Base, Self::Base, BigInt, BigInt) {
    // Dummy parameters - this group doesn't support MSMs
    (
      F::ZERO,                          // A coefficient
      F::ZERO,                          // B coefficient
      BigInt::from(GOLDILOCKS_MODULUS), // Group order
      BigInt::from(GOLDILOCKS_MODULUS), // Base field size
    )
  }
}

impl<G: Group> TranscriptReprTrait<G> for F {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_repr().to_vec()
  }
}

impl<G: Group> TranscriptReprTrait<G> for UnitPoint {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    vec![0u8; 32] // Dummy representation
  }
}

/// Re-exports that give access to the standard aliases used in the code base
pub mod goldi {
  pub use super::{F, UnitPoint};
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;

  #[test]
  fn test_basic_arithmetic() {
    let a = F::from(100);
    let b = F::from(200);

    assert_eq!(a + b, F::from(300));
    assert_eq!(b - a, F::from(100));
    assert_eq!(a * b, F::from(20000));

    // Test modular reduction
    let large = F::from(GOLDILOCKS_MODULUS - 1);
    assert_eq!(large + F::ONE, F::ZERO);
  }

  #[test]
  fn test_inversion() {
    let a = F::from(7);
    let inv = a.invert().unwrap();
    assert_eq!(a * inv, F::ONE);
  }

  #[test]
  fn test_serialization() {
    let a = F::from(12345);
    let repr = a.to_repr();
    let b = F::from_repr(repr).unwrap();
    assert_eq!(a, b);
  }
}
