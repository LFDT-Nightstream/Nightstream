//! Typed parameter sets for Neo (Nguyen–Setty 2025).
//!
//! Exposes field/cyclotomic/commitment/folding parameters and enforces:
//!  1) (k+1)·T·(b−1) < B where B=b^k  [Π_RLC bound]
//!  2) Extension policy v1 for sum-check soundness:
//!     - s_min = ceil((λ + log2(ℓ·d_sc)) / log2(q))
//!     - support only s=2; if s_min>2, return a configuration error
//!     - and record slack_bits when s_min ≤ 2.
//!
//! Symbols match the paper: q, η, d=φ(η), κ (kappa), m, b, k, B, T, s.
//!
//! References: Sec. 3–4 (Ajtai, strong set, Π_RLC bound); Sec. 6.2 (GL preset).
//!
//! NOTE: The per-instance (ℓ, d_sc) used in the sum-check live in neo-fold.
//!       Use `extension_check()` *there* with the preset's q, s, λ.

use core::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_snake_case)] // Allow mathematical notation from paper (B, T, etc.)
pub struct NeoParams {
    /// Base field modulus q (e.g., Goldilocks q = 2^64 − 2^32 + 1).
    pub q: u64,
    /// Cyclotomic index η (e.g., 81). φ(η) = d is the ring/coefficient dimension.
    pub eta: u32,
    /// d = φ(η) (e.g., 54 when η=81). Also the dimension of S⊂F_q^{d×d}.
    pub d: u32,
    /// MSIS module rank κ used in Ajtai Setup(M ∈ R_q^{κ×m}).
    pub kappa: u32,
    /// Number of columns (message length) m committed with Ajtai.
    pub m: u64,
    /// Decomposition base b (usually 2).
    pub b: u32,
    /// Folding exponent k so that B = b^k.
    pub k: u32,
    /// Upper ℓ∞ bound used by Ajtai binding *after* RLC: B = b^k.
    pub B: u64,
    /// Expansion factor of the strong challenge set C ⊂ S (empirical/spec bound).
    pub T: u32,
    /// Extension degree s used by sum-check over K=F_{q^s} (v1 supports s=2 only).
    pub s: u32,
    /// Target soundness λ in bits for the sum-check (e.g., 128).
    pub lambda: u32,
}

/// Summary returned by the extension policy check for a given (ℓ, d_sc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtensionSummary {
    pub s_min: u32,
    pub s_supported: u32,
    /// slack_bits = s_supported·log2(q) − (λ + log2(ℓ·d_sc))
    pub slack_bits: i32,
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum ParamsError {
    #[error("invalid parameter: {0}")]
    Invalid(&'static str),
    #[error("guard violated: (k+1)·T·(b−1) < B fails")]
    GuardInequality,
    #[error("unsupported extension degree; required s={required}, supported s=2")]
    UnsupportedExtension { required: u32 },
}

impl NeoParams {
    /// Construct and validate a parameter set; computes B=b^k and enforces the RLC guard.
    #[allow(non_snake_case)] // Allow mathematical notation from paper
    #[allow(clippy::too_many_arguments)] // All parameters needed for comprehensive validation
    pub fn new(
        q: u64,
        eta: u32,
        d: u32,
        kappa: u32,
        m: u64,
        b: u32,
        k: u32,
        T: u32,
        s: u32,
        lambda: u32,
    ) -> Result<Self, ParamsError> {
        if q == 0 { return Err(ParamsError::Invalid("q must be nonzero")); }
        if eta == 0 { return Err(ParamsError::Invalid("eta must be > 0")); }
        if d == 0  { return Err(ParamsError::Invalid("d must be > 0")); }
        if kappa == 0 { return Err(ParamsError::Invalid("kappa must be > 0")); }
        if m == 0 { return Err(ParamsError::Invalid("m must be > 0")); }
        if b < 2 { return Err(ParamsError::Invalid("b must be >= 2")); }
        if k == 0 { return Err(ParamsError::Invalid("k must be > 0")); }
        if T == 0 { return Err(ParamsError::Invalid("T must be > 0")); }
        if s != 2 { return Err(ParamsError::UnsupportedExtension { required: s }); } // v1 policy
        if lambda == 0 { return Err(ParamsError::Invalid("lambda must be > 0")); }

        let B = pow_u64_checked(b as u64, k)?;
        // Enforce (k+1)·T·(b-1) < B   [Π_RLC bound]
        let lhs = (k as u128 + 1) * (T as u128) * ((b as u128).saturating_sub(1));
        if lhs >= (B as u128) {
            return Err(ParamsError::GuardInequality);
        }

        Ok(Self { q, eta, d, kappa, m, b, k, B, T, s, lambda })
    }

    /// Goldilocks (~128-bit), Section 6.2: η=81, d=54, κ=16, m=2^24, b=2, k=12, B=4096, T≈216, s=2.
    /// Guard: (k+1)T(b−1)=13·216·1=2808 < 4096 ✓
    #[allow(non_snake_case)] // Allow mathematical notation from paper
    pub fn goldilocks_128() -> Self {
        // Values from the paper; see Sec. 6.2.  K = F_{q^2}.
        // q = 2^64 − 2^32 + 1 = 0xFFFFFFFF00000001
        let q: u64 = 0xFFFF_FFFF_0000_0001;
        let eta: u32 = 81;
        let d: u32 = 54;
        let kappa: u32 = 16;
        let m: u64 = 1u64 << 24;
        let b: u32 = 2;
        let k: u32 = 12;
        let T: u32 = 216;
        let s: u32 = 2;
        let lambda: u32 = 128;

        // new() computes/validates B and guard; unwrap() is safe for a known-good preset.
        Self::new(q, eta, d, kappa, m, b, k, T, s, lambda).unwrap()
    }

    /// Compute the minimal extension degree for given (ℓ, d_sc) under target λ.
    /// s_min = ceil( (λ + log2(ℓ·d_sc)) / log2(q) )
    pub fn s_min(&self, ell: u32, d_sc: u32) -> u32 {
        let ld = (ell as u128) * (d_sc as u128);
        let num = (self.lambda as f64) + log2_u128(ld);
        let den = (self.q as f64).log2();
        (num / den).ceil() as u32
    }

    /// Extension policy v1: support s=2 only. If s_min>2, return UnsupportedExtension{required=s_min}.
    /// When s_min ≤ 2, return slack_bits = floor( 2·log2(q) − (λ + log2(ℓ·d_sc)) ).
    pub fn extension_check(&self, ell: u32, d_sc: u32) -> Result<ExtensionSummary, ParamsError> {
        let s_min = self.s_min(ell, d_sc);
        if s_min > 2 {
            return Err(ParamsError::UnsupportedExtension { required: s_min });
        }
        let slack = (2.0 * (self.q as f64).log2()) - ((self.lambda as f64) + log2_u128((ell as u128) * (d_sc as u128)));
        Ok(ExtensionSummary {
            s_min,
            s_supported: 2,
            slack_bits: slack.floor() as i32,
        })
    }
}

// ---------- small helpers ----------

fn pow_u64_checked(base: u64, mut exp: u32) -> Result<u64, ParamsError> {
    let mut acc: u128 = 1;
    let mut b: u128 = base as u128;
    while exp > 0 {
        if (exp & 1) == 1 { acc = acc.checked_mul(b).ok_or(ParamsError::Invalid("B overflow"))?; }
        exp >>= 1;
        if exp > 0 { b = b.checked_mul(b).ok_or(ParamsError::Invalid("B overflow"))?; }
    }
    acc.try_into().map_err(|_| ParamsError::Invalid("B overflow"))
}

fn log2_u128(x: u128) -> f64 {
    if x == 0 { return f64::NEG_INFINITY; }
    // exact for powers of two; close enough elsewhere for s_min
    (128 - x.leading_zeros() as i32 - 1) as f64 + {
        let top = 1u128 << (127 - x.leading_zeros() as i32);
        ((x as f64) / (top as f64)).log2()
    }
}

impl fmt::Display for NeoParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NeoParams{{ q=0x{:016X}, η={}, d={}, κ={}, m={}, b={}, k={}, B={}, T={}, s={}, λ={} }}",
            self.q, self.eta, self.d, self.kappa, self.m, self.b, self.k, self.B, self.T, self.s, self.lambda
        )
    }
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goldilocks_128_matches_guard_and_b() {
        let p = NeoParams::goldilocks_128();
        assert_eq!(p.B, 4096);
        let lhs = (p.k as u128 + 1) * (p.T as u128) * ((p.b as u128) - 1);
        assert!(lhs < p.B as u128, "guard must hold");
    }

    #[test]
    fn s_min_monotone_in_lambda() {
        let p = NeoParams::goldilocks_128();
        // Pick a modest (ℓ, d_sc) representative for small CCS polynomials
        let (ell, d_sc) = (32u32, 8u32);
        // With λ=128 in this synthetic setting, s_min may be ≥2; check monotonicity only.
        let s1 = p.s_min(ell, d_sc);
        let mut tighter = p; tighter.lambda = 192;
        let s2 = tighter.s_min(ell, d_sc);
        assert!(s2 >= s1);
    }

    #[test]
    fn extension_policy_enforces_s_eq_2() {
        let mut p = NeoParams::goldilocks_128();
        // s!=2 not supported
        p.s = 3;
        assert_eq!(
            Err(ParamsError::UnsupportedExtension { required: 3 }),
            NeoParams::new(p.q, p.eta, p.d, p.kappa, p.m, p.b, p.k, p.T, 3, p.lambda)
        );
    }

    #[test]
    fn serde_roundtrip() {
        let p = NeoParams::goldilocks_128();
        let s = serde_json::to_string(&p).unwrap();
        let back: NeoParams = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
    }
}