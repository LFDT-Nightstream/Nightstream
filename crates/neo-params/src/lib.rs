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
//!
//! ## Cryptographic Primitives
//!
//! This crate also provides the canonical Poseidon2 configuration used throughout Neo.
//! All hash operations (transcripts, digests) MUST use this single source of truth.

use core::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Production Poseidon2 over Goldilocks (single source of truth)
pub mod poseidon2_goldilocks;

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
    /// Related to decomposition exponent where B = b^{k_rho}.
    pub k_rho: u32,
    /// Upper ℓ∞ bound used by Ajtai binding *after* RLC: B = b^{k_rho}.
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
    /// This includes cases where 2^λ * (ℓ·d_sc) overflows u128, implying s_min ≥ 3.
    UnsupportedExtension { required: u32 },
}

impl NeoParams {
    /// Construct and validate a parameter set; computes B=b^{k_rho} and enforces the RLC guard.
    #[allow(non_snake_case)] // Allow mathematical notation from paper
    #[allow(clippy::too_many_arguments)] // All parameters needed for comprehensive validation
    pub fn new(
        q: u64,
        eta: u32,
        d: u32,
        kappa: u32,
        m: u64,
        b: u32,
        k_rho: u32,
        T: u32,
        s: u32,
        lambda: u32,
    ) -> Result<Self, ParamsError> {
        if q == 0 {
            return Err(ParamsError::Invalid("q must be nonzero"));
        }
        if eta == 0 {
            return Err(ParamsError::Invalid("eta must be > 0"));
        }
        if d == 0 {
            return Err(ParamsError::Invalid("d must be > 0"));
        }
        if kappa == 0 {
            return Err(ParamsError::Invalid("kappa must be > 0"));
        }
        if m == 0 {
            return Err(ParamsError::Invalid("m must be > 0"));
        }
        if b < 2 {
            return Err(ParamsError::Invalid("b must be >= 2"));
        }
        if k_rho == 0 {
            return Err(ParamsError::Invalid("k_rho must be > 0"));
        }
        if T == 0 {
            return Err(ParamsError::Invalid("T must be > 0"));
        }
        if s != 2 {
            return Err(ParamsError::UnsupportedExtension { required: s });
        } // v1 policy
        if lambda == 0 {
            return Err(ParamsError::Invalid("lambda must be > 0"));
        }

        let B = pow_u64_checked(b as u64, k_rho)?;
        // Enforce (k_rho+1)·T·(b-1) < B   [Π_RLC bound]
        let lhs = (k_rho as u128 + 1) * (T as u128) * ((b as u128).saturating_sub(1));
        if lhs >= (B as u128) {
            return Err(ParamsError::GuardInequality);
        }

        Ok(Self {
            q,
            eta,
            d,
            kappa,
            m,
            b,
            k_rho,
            B,
            T,
            s,
            lambda,
        })
    }

    /// Goldilocks (~127-bit), Section 6.2: η=81, d=54, κ=16, m=2^24, b=2, k_rho=12, B=4096, T≈216, s=2.
    /// With Goldilocks q = 2^64 - 2^32 + 1, log₂(q) ≈ 63.999999999966 < 64, so q² < 2^128.
    /// For s=2 to be viable, we target λ=127 bits, giving ~127.999 bits of actual security.
    /// Guard: (k_rho+1)T(b−1)=13·216·1=2808 < 4096 ✓
    #[allow(non_snake_case)] // Allow mathematical notation from paper
    pub fn goldilocks_127() -> Self {
        // Values from the paper; see Sec. 6.2.  K = F_{q^2}.
        // q = 2^64 − 2^32 + 1 = 0xFFFFFFFF00000001
        let q: u64 = 0xFFFF_FFFF_0000_0001;
        let eta: u32 = 81;
        let d: u32 = 54;
        let kappa: u32 = 16;
        let m: u64 = 1u64 << 24;
        let b: u32 = 2;
        let k_rho: u32 = 12;
        let T: u32 = 216;
        let s: u32 = 2;
        let lambda: u32 = 127; // Adjusted for s=2 compatibility

        // new() computes/validates B and guard; unwrap() is safe for a known-good preset.
        Self::new(q, eta, d, kappa, m, b, k_rho, T, s, lambda).unwrap()
    }

    /// Auto-pick params for an R1CS instance reduced to CCS (Goldilocks preset).
    ///
    /// FE only needs to pass the number of R1CS constraints `n_rows`.
    /// We:
    ///   - pad `n_rows` to next power of two,
    ///   - set ℓ = ceil(log2(d * padded_rows))   // d = φ(η) from the preset
    ///   - bound d_sc for R1CS as: d_sc = 1 + max(u, 2b, 2), with u=2 (quadratic), b=preset.b
    ///     => with b=2 this gives d_sc = 5 (safe for R1CS-ish CCS)
    ///   - keep s=2 (policy v1), and search the largest λ ≤ preset λ with ≥ `safety_margin` slack.
    ///
    /// Defaults: min_lambda=96, safety_margin=2 bits. Returns UnsupportedExtension{required:3}
    /// if even λ=min_lambda would force s≥3.
    pub fn goldilocks_auto_r1cs_ccs(n_rows: usize) -> Result<Self, ParamsError> {
        Self::goldilocks_auto_r1cs_ccs_with(n_rows, 96, 2)
    }

    /// Same as above, but with explicit knobs for `min_lambda` and `safety_margin`.
    pub fn goldilocks_auto_r1cs_ccs_with(
        n_rows: usize,
        min_lambda: u32,
        safety_margin: u32,
    ) -> Result<Self, ParamsError> {
        let mut p = Self::goldilocks_127();

        // Compute (ℓ, d_sc) specialized for R1CS→CCS
        // pad rows to power of two (min 2)
        let padded_rows = if n_rows == 0 {
            2
        } else {
            n_rows.next_power_of_two().max(2)
        };
        // ℓ = ceil(log2(d * padded_rows))
        let prod: u128 = (p.d as u128) * (padded_rows as u128);
        let ell: u32 = ceil_log2_u128(prod);

        // R1CS: u = 2 (quadratic). d_sc = 1 + max(u, 2b, 2).
        let u_r1cs: u32 = 2;
        let two_b: u32 = p.b.saturating_mul(2);
        let d_sc: u32 = 1 + u_r1cs.max(two_b).max(2);

        // Search λ downward (keep s=2) until extension_check passes with slack
        let mut lam = p.lambda.max(min_lambda);
        while lam >= min_lambda {
            p.lambda = lam;
            match p.extension_check(ell, d_sc) {
                Ok(sum) if sum.slack_bits >= safety_margin as i32 => return Ok(p),
                Ok(_) | Err(ParamsError::UnsupportedExtension { .. }) => {
                    lam = lam.saturating_sub(1);
                }
                Err(e) => return Err(e),
            }
        }
        Err(ParamsError::UnsupportedExtension { required: 3 })
    }

    #[inline]
    fn bitlen_u128(x: u128) -> u32 {
        if x == 0 {
            0
        } else {
            128 - x.leading_zeros()
        }
    }

    /// Exact check for s=2: q^2 ≥ 2^λ · (ell·d_sc).
    /// Returns None if overflow prevents the check.
    fn s2_feasible(&self, ell: u32, d_sc: u32) -> Option<bool> {
        let q2 = (self.q as u128).checked_mul(self.q as u128)?; // q^2 fits for 64-bit q
        let ld = (ell as u128).checked_mul(d_sc as u128)?;
        let pow2 = 1u128.checked_shl(self.lambda)?; // None if λ ≥ 128
        let rhs = pow2.checked_mul(ld)?; // None if overflow
        Some(q2 >= rhs)
    }

    /// Compute the minimal extension degree using EXACT integer comparisons for s ∈ {1, 2}.
    /// This eliminates boundary-case optimism from bit-length ceiling approximations.
    /// Critical for soundness: bit-length methods can accept cases that actually need s=3!
    pub fn s_min(&self, ell: u32, d_sc: u32) -> u32 {
        let ld = (ell as u128) * (d_sc as u128);

        // Check s=1 exactly: q ≥ 2^λ · (ℓ·d_sc)
        if let Some(pow2) = 1u128.checked_shl(self.lambda) {
            if let Some(rhs) = pow2.checked_mul(ld) {
                if (self.q as u128) >= rhs {
                    return 1;
                }
            }
        }

        // Check s=2 exactly: q^2 ≥ 2^λ · (ℓ·d_sc)
        match self.s2_feasible(ell, d_sc) {
            Some(true) => 2,  // s=2 is sufficient
            Some(false) => 3, // s=2 insufficient, need s≥3
            None => 3,        // overflow on RHS ⇒ requires s ≥ 3
        }
    }

    /// Extension policy v1: support s=2 only. If s_min>2, return UnsupportedExtension{required=s_min}.
    /// When s_min=2, compute exact slack_bits by comparing q^2 against 2^λ·(ℓ·d_sc) directly.
    pub fn extension_check(&self, ell: u32, d_sc: u32) -> Result<ExtensionSummary, ParamsError> {
        let s_min = self.s_min(ell, d_sc);
        if s_min > 2 {
            return Err(ParamsError::UnsupportedExtension { required: s_min });
        }

        // Exact slack for s=2: compute floor(log₂(q²/(2^λ·ℓd))) without floating point
        let q = self.q as u128;
        let q2 = q * q; // q^2 cannot overflow u128 for 64-bit q
        let ld = (ell as u128).checked_mul(d_sc as u128).unwrap();

        let rhs = 1u128
            .checked_shl(self.lambda)
            .and_then(|p| p.checked_mul(ld))
            .ok_or(ParamsError::UnsupportedExtension { required: 3 })?;

        let slack_bits = if q2 < rhs {
            // This case should not happen if s_min=2, but handle gracefully
            -1
        } else {
            // Compute floor(log₂(q²/rhs)) using bit lengths
            let mut slack = Self::bitlen_u128(q2) as i32 - Self::bitlen_u128(rhs) as i32;
            // Adjust if the division has no fractional part
            if let Some(shifted) = rhs.checked_shl(slack as u32) {
                if q2 < shifted {
                    slack -= 1;
                }
            }
            slack
        };

        Ok(ExtensionSummary {
            s_min,
            s_supported: 2,
            slack_bits,
        })
    }
}

// ---------- small helpers ----------

/// ceil(log2(x)) for u128, with ceil_log2(0) = 0 and ceil_log2(1) = 0.
#[inline]
fn ceil_log2_u128(x: u128) -> u32 {
    if x <= 1 {
        0
    } else {
        128u32 - (x - 1).leading_zeros()
    }
}

fn pow_u64_checked(base: u64, mut exp: u32) -> Result<u64, ParamsError> {
    let mut acc: u128 = 1;
    let mut b: u128 = base as u128;
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = acc
                .checked_mul(b)
                .ok_or(ParamsError::Invalid("B overflow"))?;
        }
        exp >>= 1;
        if exp > 0 {
            b = b.checked_mul(b).ok_or(ParamsError::Invalid("B overflow"))?;
        }
    }
    acc.try_into()
        .map_err(|_| ParamsError::Invalid("B overflow"))
}

impl fmt::Display for NeoParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NeoParams{{ q=0x{:016X}, η={}, d={}, κ={}, m={}, b={}, k_rho={}, B={}, T={}, s={}, λ={} }}",
            self.q, self.eta, self.d, self.kappa, self.m, self.b, self.k_rho, self.B, self.T, self.s, self.lambda
        )
    }
}

// Tests live in `crates/neo-params/tests/` (no in-file test modules).
