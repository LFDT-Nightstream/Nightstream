//! `F_q` with `q = 2^64 − 59`, and `K = F_q[u]/(u² − 2)`
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

pub const Q: u64 = neo_params::nightstream_p64_k14::Q;

pub(crate) const TWO_POW_64_MOD_Q: u64 = Q.wrapping_neg();

const TWO_POW_128_MOD_Q: u64 = TWO_POW_64_MOD_Q * TWO_POW_64_MOD_Q;

const INV_7_MOD_Q_MINUS_1: u64 = 10_540_996_613_548_315_175;
const _: () = assert!((7u128 * INV_7_MOD_Q_MINUS_1 as u128) % (Q as u128 - 1) == 1);

/// An element of `F_q`. The value is always the canonical one, in `[0, q)`.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
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
        Self::reduce_folded(Self::fold128(x))
    }

    #[inline]
    const fn fold128(x: u128) -> u128 {
        (x >> 64) * TWO_POW_64_MOD_Q as u128 + (x as u64) as u128
    }

    #[inline]
    const fn reduce_folded(folded: u128) -> Self {
        let (low, carry) = (folded as u64).overflowing_add((folded >> 64) as u64 * TWO_POW_64_MOD_Q);
        let (wrapped, above) = low.overflowing_add(TWO_POW_64_MOD_Q);
        Self(if carry | above { wrapped } else { low })
    }

    /// The sum of the products `a·b`, with one reduction for all terms.
    #[inline]
    pub fn sum_of_products(pairs: impl IntoIterator<Item = (Self, Self)>) -> Self {
        pairs
            .into_iter()
            .fold(UnreducedProducts::default(), |sum, (a, b)| sum.add(a, b))
            .value()
    }
}

#[inline]
pub fn sum_of_scaled(pairs: impl IntoIterator<Item = (Fq, K)>) -> K {
    let [real, imaginary] = pairs
        .into_iter()
        .fold([UnreducedProducts::default(); 2], |[real, imaginary], (a, b)| {
            let [x, y] = b.as_coeffs();
            [real.add(a, x), imaginary.add(a, y)]
        });
    K::from_coeffs([real.value(), imaginary.value()])
}

#[derive(Clone, Copy, Default)]
pub struct ProductSum([UnreducedProducts; 2]);

const _: () = assert!(<Fq as BinomiallyExtendable<2>>::W.0 == 2);

impl ProductSum {
    #[inline]
    pub fn add(self, a: K, b: K) -> Self {
        let ([a0, a1], [b0, b1]) = (a.as_coeffs(), b.as_coeffs());
        let [real, imaginary] = self.0;
        Self([real.add(a0, b0).add(a1, b1.double()), imaginary.add(a0, b1).add(a1, b0)])
    }

    #[inline]
    pub fn add_scaled(self, a: Fq, b: K) -> Self {
        let [b0, b1] = b.as_coeffs();
        let [real, imaginary] = self.0;
        Self([real.add(a, b0), imaginary.add(a, b1)])
    }

    #[inline]
    pub fn value(self) -> K {
        let [real, imaginary] = self.0;
        K::from_coeffs([real.value(), imaginary.value()])
    }
}

#[derive(Clone, Copy, Default)]
struct UnreducedProducts {
    sum: u128,
    wraps: u64,
}

impl UnreducedProducts {
    #[inline]
    fn add(self, a: Fq, b: Fq) -> Self {
        let (sum, wrapped) = self.sum.overflowing_add(u128::from(a.0) * u128::from(b.0));
        Self {
            sum,
            wraps: self.wraps + u64::from(wrapped),
        }
    }

    #[inline]
    fn value(self) -> Fq {
        Fq::reduce128(self.sum) + Fq::reduce128(u128::from(self.wraps) * u128::from(TWO_POW_128_MOD_Q))
    }
}

#[derive(Clone, Copy, Default)]
pub struct CompactProductSum([[u64; 3]; 2]);

impl CompactProductSum {
    #[inline]
    pub fn add_scaled_word(self, a: u64, b: K) -> Self {
        let terms = b.as_coeffs();
        Self(std::array::from_fn(|i| {
            let [lo, hi, wraps] = self.0[i];
            let sum = (u128::from(hi) << 64) | u128::from(lo);
            let (sum, wrapped) = sum.overflowing_add(u128::from(a) * u128::from(terms[i].0));
            [sum as u64, (sum >> 64) as u64, wraps + u64::from(wrapped)]
        }))
    }

    #[inline]
    pub fn value(self) -> K {
        K::from_coeffs(self.0.map(|[lo, hi, wraps]| {
            UnreducedProducts {
                sum: (u128::from(hi) << 64) | u128::from(lo),
                wraps,
            }
            .value()
        }))
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

/// `gcd(7, q − 1) = 1`, so `x ↦ x^7` is a permutation (Lean
/// `Spec.Poseidon2.coprime_seven`, `Spec.Poseidon2.sbox_bijective`).
impl InjectiveMonomial<7> for Fq {}

impl PermutationMonomial<7> for Fq {
    fn injective_exp_root_n(&self) -> Self {
        self.exp_u64(INV_7_MOD_Q_MINUS_1)
    }
}

impl RawDataSerializable for Fq {
    impl_raw_serializable_primefield64!();
}

impl Packable for Fq {}

impl Field for Fq {
    type Packing = Self;

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

impl BinomiallyExtendableAlgebra<Self, 2> for Fq {
    #[inline]
    fn binomial_mul(&[a0, a1]: &[Self; 2], &[b0, b1]: &[Self; 2], res: &mut [Self; 2], w: Self) {
        let fold_product = |a: Self, b: Self| Self::fold128(u128::from(a.0) * u128::from(b.0));
        let quadratic = if w == Self::TWO {
            2 * fold_product(a1, b1)
        } else {
            fold_product(a1, b1 * w)
        };
        *res = [
            Self::reduce_folded(fold_product(a0, b0) + quadratic),
            Self::reduce_folded(fold_product(a0, b1) + fold_product(a1, b0)),
        ];
    }
}

impl BinomiallyExtendable<2> for Fq {
    /// 2 is not a square modulo `q`, so `u² − 2` is irreducible (Lean
    /// `Spec.FieldExtension.twoProjectiveNonresidue`).
    const W: Self = Self::TWO;

    const DTH_ROOT: Self = Self::NEG_ONE;

    const EXT_GENERATOR: [Self; 2] = [Self(3), Self::ONE];
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
