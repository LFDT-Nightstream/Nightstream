//! Goldilocks base-field and K = F_{q^2} arithmetic usable from device code.
//!
//! Contract: `Gl` must match `neo_math::F` (Goldilocks, q = 2^64 - 2^32 + 1)
//! and `Kx` must match `neo_math::K` (quadratic extension, u^2 = 7) exactly;
//! the `parity smoke` gate asserts both against the CPU implementations.
//! Values are canonical u64 words at every API boundary.

pub const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;
const GOLDILOCKS_EPSILON: u64 = 0xffff_ffff; // 2^64 mod q

/// Goldilocks base-field element, always held in canonical form.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Gl(u64);

impl Gl {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    pub fn from_u64(value: u64) -> Self {
        if value >= GOLDILOCKS_MODULUS {
            Self(value - GOLDILOCKS_MODULUS)
        } else {
            Self(value)
        }
    }

    pub fn as_canonical_u64(self) -> u64 {
        self.0
    }
}

impl core::ops::Add for Gl {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let (sum, carry) = self.0.overflowing_add(rhs.0);
        let mut out = if carry {
            sum.wrapping_add(GOLDILOCKS_EPSILON)
        } else {
            sum
        };
        if out >= GOLDILOCKS_MODULUS {
            out -= GOLDILOCKS_MODULUS;
        }
        Self(out)
    }
}

impl core::ops::Sub for Gl {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let (diff, borrow) = self.0.overflowing_sub(rhs.0);
        if borrow {
            Self(diff.wrapping_sub(GOLDILOCKS_EPSILON))
        } else {
            Self(diff)
        }
    }
}

impl core::ops::Mul for Gl {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(reduce_u128((self.0 as u128) * (rhs.0 as u128)))
    }
}

impl core::ops::Neg for Gl {
    type Output = Self;

    fn neg(self) -> Self {
        Self::ZERO - self
    }
}

/// `a · z` where `z` is expected to be a balanced low-norm value: canonical
/// form is either a small magnitude or `q − magnitude`. Splits off the sign,
/// multiplies by the ≤32-bit magnitude with two 32×32 products (cheaper than
/// the emulated 64×64), and negates back — falling back to the full multiply
/// whenever the magnitude exceeds 32 bits, so the result is value-equal to
/// `a * Gl::from_u64(z)` for every input.
pub fn mul_low_norm(a: Gl, z: u64) -> Gl {
    let neg = z > GOLDILOCKS_MODULUS / 2;
    let m = if neg { GOLDILOCKS_MODULUS - z } else { z };
    if (m >> 32) != 0 {
        return a * Gl::from_u64(z);
    }
    let p0 = ((a.0 & 0xffff_ffff) as u32 as u64) * (m as u32 as u64);
    let p1 = ((a.0 >> 32) as u32 as u64) * (m as u32 as u64);
    let out = Gl(reduce_u128((p0 as u128) + ((p1 as u128) << 32)));
    if neg {
        -out
    } else {
        out
    }
}

/// Reduce a 192-bit value `lo + mid·2^64 + hi·2^128` mod q via two exact
/// folds: `X = (hi·2^64 + mid) mod q`, then `(X·2^64 + lo) mod q`. Used by
/// lazy product accumulation (sum raw 128-bit products, reduce once).
pub fn reduce_192(lo: u64, mid: u64, hi: u64) -> Gl {
    let x = reduce_u128(((hi as u128) << 64) | (mid as u128));
    Gl(reduce_u128(((x as u128) << 64) | (lo as u128)))
}

fn reduce_u128(value: u128) -> u64 {
    // Write the high word as `hi_hi * 2^32 + hi_lo`. Since
    // `2^64 = 2^32 - 1 (mod q)` and
    // `2^32 * (2^32 - 1) = -1 (mod q)`, one exact fold is
    //
    //   lo + hi_lo * (2^32 - 1) - hi_hi.
    //
    // Every term fits in u64; the Gl operations perform the two modular
    // corrections without an emulated-u128 reduction loop.
    let lo = value as u64;
    let hi = (value >> 64) as u64;
    let hi_hi = hi >> 32;
    let hi_lo = hi & GOLDILOCKS_EPSILON;
    let (diff, borrow) = lo.overflowing_sub(hi_hi);
    let reduced_lo = if borrow {
        diff.wrapping_sub(GOLDILOCKS_EPSILON)
    } else {
        diff
    };
    let folded_hi = (hi_lo << 32).wrapping_sub(hi_lo);
    let (sum, carry) = reduced_lo.overflowing_add(folded_hi);
    let mut out = if carry {
        sum.wrapping_add(GOLDILOCKS_EPSILON)
    } else {
        sum
    };
    if out >= GOLDILOCKS_MODULUS {
        out -= GOLDILOCKS_MODULUS;
    }
    out
}

/// K = F_{q^2} element `c0 + c1·u` with `u^2 = 7`, matching `neo_math::K`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Kx {
    c0: Gl,
    c1: Gl,
}

impl Kx {
    pub const ZERO: Self = Self {
        c0: Gl::ZERO,
        c1: Gl::ZERO,
    };
    pub const ONE: Self = Self {
        c0: Gl::ONE,
        c1: Gl::ZERO,
    };

    pub fn from_components(c0: Gl, c1: Gl) -> Self {
        Self { c0, c1 }
    }

    pub fn from_words(c0: u64, c1: u64) -> Self {
        Self {
            c0: Gl::from_u64(c0),
            c1: Gl::from_u64(c1),
        }
    }

    pub fn as_words(self) -> [u64; 2] {
        [self.c0.as_canonical_u64(), self.c1.as_canonical_u64()]
    }

    pub fn scale_base(self, scalar: Gl) -> Self {
        Self {
            c0: self.c0 * scalar,
            c1: self.c1 * scalar,
        }
    }
}

impl core::ops::Add for Kx {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
        }
    }
}

impl core::ops::Sub for Kx {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
        }
    }
}

impl core::ops::Mul for Kx {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 * rhs.c0 + (self.c1 * rhs.c1) * Gl::from_u64(7),
            c1: self.c0 * rhs.c1 + self.c1 * rhs.c0,
        }
    }
}

impl core::ops::Neg for Kx {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
        }
    }
}
