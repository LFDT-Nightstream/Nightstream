use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Trait for coefficient types in polys/rings (shared with neo-poly).
use rand::Rng;

pub trait Coeff:
    Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Neg<Output = Self>
    + Copy
    + PartialEq
    + Eq
    + Debug
    + Default
{
    fn zero() -> Self;
    fn one() -> Self;

    /// Return the modulus of the underlying ring/field.
    fn modulus() -> u64;

    /// Sample a random element uniformly from the modulus.
    fn random(rng: &mut impl Rng) -> Self;

    /// Multiplicative inverse (assumes field).
    fn inverse(&self) -> Self;
}

/// Modular integer over prime q (lattice modulus).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ModInt {
    val: u64,
}

impl ModInt {
    /// Lattice modulus from paper (Mersenne prime).
    pub const Q: u64 = (1u64 << 61) - 1;

    pub fn from_u64(val: u64) -> Self {
        Self { val: val % Self::Q }
    }

    pub fn as_u64(&self) -> u64 {
        self.val
    }

    /// Return the canonical u64 representation of this field element.
    ///
    /// This is equivalent to [`as_u64`] since `ModInt` already stores a
    /// representative in the range `[0, Q)`.
    pub fn as_canonical_u64(&self) -> u64 {
        self.as_u64()
    }

    pub fn inverse(&self) -> Self {
        if self.val == 0 {
            panic!("inverse of zero");
        }
        let (gcd, x, _) = extended_euclid(self.val as i128, Self::Q as i128);
        assert_eq!(gcd, 1);
        let q = Self::Q as i128;
        let inv = ((x % q) + q) % q;
        Self { val: inv as u64 }
    }
}

fn extended_euclid(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (gcd, x1, y1) = extended_euclid(b % a, a);
        (gcd, y1 - (b / a) * x1, x1)
    }
}

impl Add for ModInt {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut sum = self.val + rhs.val;
        if sum >= Self::Q {
            sum -= Self::Q;
        }
        Self { val: sum }
    }
}

impl Sub for ModInt {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut diff = self.val.wrapping_sub(rhs.val);
        if diff > Self::Q {
            diff = diff.wrapping_add(Self::Q);
        }
        Self { val: diff }
    }
}

impl Mul for ModInt {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let prod = (self.val as u128 * rhs.val as u128) % (Self::Q as u128);
        Self { val: prod as u64 }
    }
}

impl Neg for ModInt {
    type Output = Self;
    fn neg(self) -> Self {
        if self.val == 0 {
            self
        } else {
            Self {
                val: Self::Q - self.val,
            }
        }
    }
}

impl AddAssign for ModInt {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for ModInt {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for ModInt {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl From<u64> for ModInt {
    fn from(val: u64) -> Self {
        Self::from_u64(val)
    }
}

impl From<i128> for ModInt {
    fn from(val: i128) -> Self {
        let q = Self::Q as i128;
        let mut v = val % q;
        if v < 0 {
            v += q;
        }
        Self::from_u64(v as u64)
    }
}

impl From<ModInt> for u64 {
    fn from(val: ModInt) -> u64 {
        val.val
    }
}

impl From<ModInt> for i128 {
    fn from(val: ModInt) -> i128 {
        let q = ModInt::Q as i128;
        let val_i128 = val.val as i128;
        if val_i128 > q / 2 {
            val_i128 - q
        } else {
            val_i128
        }
    }
}

#[cfg(feature = "quickcheck")]
impl Arbitrary for ModInt {
    fn arbitrary(g: &mut Gen) -> Self {
        let val = u64::arbitrary(g) % Self::Q;
        Self::from_u64(val)
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        Box::new(std::iter::empty())
    }
}

impl PartialOrd for ModInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ModInt {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.val.cmp(&other.val)
    }
}

impl Coeff for ModInt {
    fn zero() -> Self {
        ModInt { val: 0 }
    }
    fn one() -> Self {
        ModInt { val: 1 }
    }

    fn modulus() -> u64 {
        Self::Q
    }

    fn random(rng: &mut impl Rng) -> Self {
        let val = rng.random::<u64>() % Self::Q;
        Self::from_u64(val)
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }
}

use neo_fields::F;
use p3_field::{integers::QuotientMap, Field, PrimeCharacteristicRing, PrimeField64};
#[cfg(feature = "quickcheck")]
use quickcheck::{Arbitrary, Gen};

impl Coeff for F {
    fn zero() -> Self {
        F::ZERO
    }
    fn one() -> Self {
        F::ONE
    }

    fn modulus() -> u64 {
        <F as PrimeField64>::ORDER_U64
    }

    fn random(rng: &mut impl Rng) -> Self {
        let val = rng.random::<u64>() % <F as PrimeField64>::ORDER_U64;
        <F as QuotientMap<u64>>::from_int(val)
    }

    fn inverse(&self) -> Self {
        Field::inverse(self)
    }
}

use neo_fields::ExtF;

impl Coeff for ExtF {
    fn zero() -> Self {
        ExtF::ZERO
    }
    fn one() -> Self {
        ExtF::ONE
    }

    fn modulus() -> u64 {
        <F as PrimeField64>::ORDER_U64
    }

    fn random(rng: &mut impl Rng) -> Self {
        neo_fields::random_extf(rng)
    }

    fn inverse(&self) -> Self {
        Field::inverse(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ops() {
        let a = ModInt::from_u64(5);
        let b = ModInt::from_u64(3);
        assert_eq!(a + b, ModInt::from_u64(8));
        assert_eq!(a - b, ModInt::from_u64(2));
        assert_eq!(a * b, ModInt::from_u64(15));
        assert_eq!(-a, ModInt::from_u64(ModInt::Q - 5));
    }

    #[test]
    fn test_overflow() {
        let max = ModInt::from_u64(ModInt::Q - 1);
        assert_eq!(max + ModInt::one(), ModInt::zero());
        assert_eq!(
            ModInt::zero() - ModInt::one(),
            ModInt::from_u64(ModInt::Q - 1)
        );
    }

    #[test]
    fn test_inverse() {
        let a = ModInt::from_u64(5);
        let inv = a.inverse();
        assert_eq!(a * inv, ModInt::one());
    }
}
