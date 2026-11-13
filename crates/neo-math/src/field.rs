//! Field layer: `F_q` with `q = 2^64 − 59`, and `K = F_q[u]/(u² − 7)`
//! (`decisions/field-2p64-59-ring-x64.md`).

use core::array;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable, BinomiallyExtendableAlgebra};
use p3_field::integers::QuotientMap;
use p3_field::{
    impl_raw_serializable_primefield64, Field, InjectiveMonomial, Packable, PermutationMonomial,
    PrimeCharacteristicRing, PrimeField, PrimeField64, RawDataSerializable,
};
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};

pub const Q: u64 = neo_params::nightstream_p64_k16::Q;

/// `2^64 mod q`.
pub(crate) const FOLD: u64 = Q.wrapping_neg();

/// An element of `F_q`. The value is always the canonical one, in `[0, q)`.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fq(u64);

pub type K = BinomialExtensionField<Fq, 2>;

impl Fq {
    /// The element `value mod q`.
    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(if value >= Q { value - Q } else { value })
    }

    #[inline]
    const fn reduce128(x: u128) -> Self {
        // 2^64 ≡ 59 (mod q), so hi·2^64 + lo ≡ hi·59 + lo, which is below 2^70.
        let folded = (x >> 64) * FOLD as u128 + (x as u64) as u128;
        // The high part of `folded` is below 60, so the addend is below 3540.
        let (low, carry) = (folded as u64).overflowing_add((folded >> 64) as u64 * FOLD);
        // A carry dropped 2^64 ≡ 59, and then `low` is below 3540. Without a
        // carry, `low + 59` wraps exactly when `low ≥ q`, and it wraps to `low − q`.
        let (wrapped, above) = low.overflowing_add(FOLD);
        Self(if carry | above { wrapped } else { low })
    }

    /// The sum of the products `a·b`, with one reduction for all terms.
    #[inline]
    pub fn sum_of_products(pairs: impl IntoIterator<Item = (Self, Self)>) -> Self {
        let (sum, wraps) = pairs
            .into_iter()
            .fold((0u128, 0u64), |(sum, wraps), (a, b)| {
                let (sum, wrapped) = sum.overflowing_add(u128::from(a.0) * u128::from(b.0));
                (sum, wraps + u64::from(wrapped))
            });
        // Each wrap dropped 2^128 ≡ 59² (mod q).
        Self::reduce128(sum) + Self::reduce128(u128::from(wraps) * u128::from(FOLD * FOLD))
    }
}

impl Display for Fq {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Debug for Fq {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Serialize for Fq {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for Fq {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = u64::deserialize(deserializer)?;
        Self::from_canonical_checked(value).ok_or_else(|| D::Error::custom("field element is not below the modulus"))
    }
}

impl Add for Fq {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (sum, carry) = self.0.overflowing_add(rhs.0);
        let (reduced, borrow) = sum.overflowing_sub(Q);
        Self(if carry || !borrow { reduced } else { sum })
    }
}

impl Sub for Fq {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (difference, borrow) = self.0.overflowing_sub(rhs.0);
        Self(if borrow { difference.wrapping_add(Q) } else { difference })
    }
}

impl Neg for Fq {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::ZERO - self
    }
}

impl Mul for Fq {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::reduce128(u128::from(self.0) * u128::from(rhs.0))
    }
}

impl Div for Fq {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

macro_rules! assign_ops {
    ($($trait:ident $method:ident $op:tt),*) => {$(
        impl $trait for Fq {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                *self = *self $op rhs;
            }
        }
    )*};
}
assign_ops!(AddAssign add_assign +, SubAssign sub_assign -, MulAssign mul_assign *, DivAssign div_assign /);

impl Sum for Fq {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::reduce128(iter.map(|x| u128::from(x.0)).sum())
    }
}

impl Product for Fq {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |product, x| product * x)
    }
}

impl PrimeCharacteristicRing for Fq {
    type PrimeSubfield = Self;

    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const TWO: Self = Self(2);
    const NEG_ONE: Self = Self(Q - 1);

    #[inline]
    fn from_prime_subfield(f: Self) -> Self {
        f
    }

    #[inline]
    fn halve(&self) -> Self {
        let half = self.0 >> 1;
        Self(if self.0 & 1 == 0 { half } else { half + Q.div_ceil(2) })
    }

    #[inline]
    fn dot_product<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        Self::sum_of_products(lhs.iter().copied().zip(rhs.iter().copied()))
    }
}

/// `gcd(7, q − 1) = 1`, so `x ↦ x^7` is a permutation.
impl InjectiveMonomial<7> for Fq {}

impl PermutationMonomial<7> for Fq {
    fn injective_exp_root_n(&self) -> Self {
        // 7 · 10540996613548315175 ≡ 1 (mod q − 1).
        self.exp_u64(10_540_996_613_548_315_175)
    }
}

impl RawDataSerializable for Fq {
    impl_raw_serializable_primefield64!();
}

impl Packable for Fq {}

impl Field for Fq {
    type Packing = Self;

    /// 2 generates `F_q*`; `q − 1 = 2² · 11 · 137 · 547 · 5594472617641`.
    const GENERATOR: Self = Self::TWO;

    fn try_inverse(&self) -> Option<Self> {
        (self.0 != 0).then(|| self.exp_u64(Q - 2))
    }

    fn order() -> BigUint {
        Q.into()
    }
}

/// `QuotientMap` declares `from_canonical_unchecked` as an `unsafe fn`. The
/// bodies here are safe: they reduce their input.
macro_rules! quotient_map {
    ($kind:ident: $($int:ty),*) => {$(
        #[allow(unsafe_code)]
        impl QuotientMap<$int> for Fq {
            #[inline]
            fn from_int(int: $int) -> Self {
                quotient_map!(@reduce $kind int)
            }

            #[inline]
            fn from_canonical_checked(int: $int) -> Option<Self> {
                quotient_map!(@canonical $kind int).then(|| Self::from_int(int))
            }

            #[inline]
            unsafe fn from_canonical_unchecked(int: $int) -> Self {
                Self::from_int(int)
            }
        }
    )*};
    (@reduce unsigned $int:ident) => {
        Self::reduce128($int as u128)
    };
    (@reduce signed $int:ident) => {{
        let magnitude = Self::reduce128($int.unsigned_abs() as u128);
        if $int < 0 { -magnitude } else { magnitude }
    }};
    (@canonical unsigned $int:ident) => {
        ($int as u128) < Q as u128
    };
    (@canonical signed $int:ident) => {
        $int.unsigned_abs() as u128 <= (Q as u128) / 2
    };
}
quotient_map!(unsigned: u8, u16, u32, u64, u128);
quotient_map!(signed: i8, i16, i32, i64, i128);

impl PrimeField for Fq {
    fn as_canonical_biguint(&self) -> BigUint {
        self.0.into()
    }
}

impl PrimeField64 for Fq {
    const ORDER_U64: u64 = Q;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        self.0
    }
}

impl BinomiallyExtendableAlgebra<Self, 2> for Fq {}

impl BinomiallyExtendable<2> for Fq {
    /// 7 is not a square modulo `q`, so `u² − 7` is irreducible.
    const W: Self = Self(7);

    /// `W^((q − 1)/2)`.
    const DTH_ROOT: Self = Self::NEG_ONE;

    /// `2 + u` generates `K*`; `q + 1 = 2 · 3 · 7 · 439208192231179799`.
    const EXT_GENERATOR: [Self; 2] = [Self::TWO, Self::ONE];
}

pub trait KExtensions {
    fn conj(self) -> Self;
    fn inv(self) -> Self;
    fn as_coeffs(&self) -> [Fq; 2];
    fn from_coeffs(coefs: [Fq; 2]) -> Self;
    fn scale_base(self, s: Fq) -> Self;
    fn scale_base_k(self, s: Self) -> Self;
    fn real(&self) -> Fq {
        self.as_coeffs()[0]
    }
    fn imag(&self) -> Fq {
        self.as_coeffs()[1]
    }
    fn to_limbs_u64(&self) -> (u64, u64) {
        let coeffs = self.as_coeffs();
        (coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64())
    }
}

impl KExtensions for K {
    #[inline]
    fn conj(self) -> Self {
        self.conjugate()
    }
    #[inline]
    fn inv(self) -> Self {
        self.inverse()
    }
    #[inline]
    fn as_coeffs(&self) -> [Fq; 2] {
        [self.real(), self.imag()]
    }
    #[inline]
    fn from_coeffs(coefs: [Fq; 2]) -> Self {
        BinomialExtensionField::new_complex(coefs[0], coefs[1])
    }

    #[inline]
    fn scale_base(self, s: Fq) -> Self {
        let [c0, c1] = self.as_coeffs();
        Self::from_coeffs([c0 * s, c1 * s])
    }

    #[inline]
    fn scale_base_k(self, s: Self) -> Self {
        let [s0, s1] = s.as_coeffs();
        if s1 == Fq::ZERO {
            self.scale_base(s0)
        } else {
            self * s
        }
    }
}

#[inline]
pub fn from_complex(real: Fq, imag: Fq) -> K {
    K::from_coeffs([real, imag])
}
