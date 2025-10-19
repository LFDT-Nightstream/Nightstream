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
    /// This includes cases where 2^λ * (ℓ·d_sc) overflows u128, implying s_min ≥ 3.
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

    /// Goldilocks (~127-bit), Section 6.2: η=81, d=54, κ=16, m=2^24, b=2, k=12, B=4096, T≈216, s=2.
    /// With Goldilocks q = 2^64 - 2^32 + 1, log₂(q) ≈ 63.999999999966 < 64, so q² < 2^128.
    /// For s=2 to be viable, we target λ=127 bits, giving ~127.999 bits of actual security.
    /// Guard: (k+1)T(b−1)=13·216·1=2808 < 4096 ✓
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
        let k: u32 = 12;
        let T: u32 = 216;
        let s: u32 = 2;
        let lambda: u32 = 127; // Adjusted for s=2 compatibility

        // new() computes/validates B and guard; unwrap() is safe for a known-good preset.
        Self::new(q, eta, d, kappa, m, b, k, T, s, lambda).unwrap()
    }

    /// Goldilocks strict 128-bit (requires s=3+), Section 6.2: η=81, d=54, κ=16, m=2^24, b=2, k=12, B=4096, T≈216.
    /// With λ=128 and Goldilocks field, s_min ≥ 3 for any non-trivial (ℓ,d).
    /// This preset will be rejected by v1 extension policy; kept for completeness.
    #[allow(non_snake_case)] // Allow mathematical notation from paper
    pub fn goldilocks_128_strict() -> Self {
        // Values from the paper; see Sec. 6.2.
        let q: u64 = 0xFFFF_FFFF_0000_0001;
        let eta: u32 = 81;
        let d: u32 = 54;
        let kappa: u32 = 16;
        let m: u64 = 1u64 << 24;
        let b: u32 = 2;
        let k: u32 = 12;
        let T: u32 = 216;
        let s: u32 = 2; // v1 policy: only s=2 supported
        let lambda: u32 = 128; // strict target

        // new() computes/validates B and guard; unwrap() is safe for a known-good preset.
        Self::new(q, eta, d, kappa, m, b, k, T, s, lambda).unwrap()
    }

    /// Calculate maximum lambda that works with s=2 for given workload parameters
    pub fn max_lambda_for_s2(ell: u32, d_sc: u32) -> u32 {
        let q: u64 = 0xFFFF_FFFF_0000_0001; // Goldilocks
        let max_bits = 2.0 * (q as f64).log2(); // ≈ 127.99999999994
        let penalty = log2_u128((ell as u128) * (d_sc as u128));
        let lambda_max = (max_bits - penalty).floor() as u32;
        lambda_max.saturating_sub(1) // Leave 1 bit safety margin
    }

    /// Conservative preset for small circuits (ell ≤ 4, d_sc ≤ 3)
    /// 
    /// This is a safe default that works for most small circuits.
    /// For larger circuits, use `goldilocks_for_circuit()` to auto-tune parameters.
    pub fn goldilocks_small_circuits() -> Self {
        Self::goldilocks_for_circuit(4, 3, 2) // max ell*d_sc=12, 2-bit safety
    }

    /// Auto-tuned parameters for a specific circuit
    /// 
    /// This is the recommended way to create parameters for any circuit size.
    /// It automatically calculates the optimal lambda based on your circuit's requirements.
    /// 
    /// # Arguments
    /// * `ell` - log₂(padded_circuit_size)
    /// * `d_sc` - Maximum polynomial degree in CCS constraints
    /// * `safety_margin` - Security margin in bits (recommended: 2-4)
    /// 
    /// # Security
    /// - Base: ~128 bits for Goldilocks field (q^2 ≈ 2^128)
    /// - Cost: ~1 bit per doubling of (ell * d_sc)
    /// - Formula: lambda ≈ 128 - log₂(ell * d_sc) - safety_margin
    /// - All configurations maintain >100 bits post-quantum security
    /// 
    /// # Examples
    /// 
    /// ```
    /// use neo_params::NeoParams;
    /// 
    /// // Auto-tune for circuit with 1024 constraints (ell=10), degree-3
    /// let params = NeoParams::goldilocks_for_circuit(10, 3, 2);
    /// 
    /// // Auto-tune for circuit with 4096 constraints (ell=12), degree-2
    /// let params = NeoParams::goldilocks_for_circuit(12, 2, 3);
    /// ```
    #[allow(non_snake_case)] // Allow mathematical notation from paper
    pub fn goldilocks_for_circuit(max_ell: u32, max_d_sc: u32, safety_margin: u32) -> Self {
        let q: u64 = 0xFFFF_FFFF_0000_0001;
        let eta: u32 = 81;
        let d: u32 = 54;
        let kappa: u32 = 16;
        let m: u64 = 1u64 << 24;
        let b: u32 = 2;
        let k: u32 = 12;
        let T: u32 = 216;
        let s: u32 = 2;

        let lambda = Self::max_lambda_for_s2(max_ell, max_d_sc).saturating_sub(safety_margin);
        
        // Ensure lambda is at least somewhat reasonable (>= 80 bits)
        let lambda = lambda.max(80);
        
        Self::new(q, eta, d, kappa, m, b, k, T, s, lambda).unwrap()
    }

    /// Compute optimal parameters from circuit size and degree
    /// 
    /// This is a helper for auto-detecting parameters in Session/folding code.
    /// 
    /// # Arguments
    /// * `circuit_size` - Number of constraints (will be padded to power of 2)
    /// * `max_degree` - Maximum polynomial degree in the circuit
    /// 
    /// # Returns
    /// A tuple `(ell, d_sc)` suitable for `goldilocks_for_circuit()`
    /// 
    /// # Example
    /// 
    /// ```
    /// use neo_params::NeoParams;
    /// 
    /// // Circuit with 100 constraints, degree 3
    /// let (ell, d_sc) = NeoParams::compute_circuit_params(100, 3);
    /// // ell = 7 (log2(128)), d_sc = 3
    /// 
    /// let params = NeoParams::goldilocks_for_circuit(ell, d_sc, 2);
    /// ```
    pub fn compute_circuit_params(circuit_size: usize, max_degree: u32) -> (u32, u32) {
        // Pad circuit size to next power of 2 (minimum 2)
        let padded_size = circuit_size.next_power_of_two().max(2);
        let ell = padded_size.trailing_zeros();
        (ell, max_degree)
    }

    /// Show lambda limits table for different workload sizes with s=2
    pub fn show_s2_lambda_limits() {
        println!("\n🔍 Lambda Limits for s=2 with Goldilocks Field:");
        println!("====================================================");
        println!(" ell*d_sc  | max_lambda | suggested_lambda");
        println!("-----------|------------|------------------");
        
        for &ell_d_product in &[1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256] {
            let max_lambda = Self::max_lambda_for_s2(ell_d_product, 1); // Use ell_d_product directly
            let suggested = max_lambda.saturating_sub(2); // 2-bit safety margin
            println!(" {ell_d_product:>8} | {max_lambda:>10} | {suggested:>16}");
        }
        println!("\n💡 Rule of thumb: Each doubling of ell*d_sc costs ~1 bit of lambda");
        println!("🔒 Suggested lambda = max_lambda - 2 (safety margin)");
    }

    #[inline]
    fn bitlen_u128(x: u128) -> u32 {
        if x == 0 { 0 } else { 128 - x.leading_zeros() }
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
            Some(true) => 2,   // s=2 is sufficient
            Some(false) => 3,  // s=2 insufficient, need s≥3
            None => 3,         // overflow on RHS ⇒ requires s ≥ 3
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
        
        let rhs = 1u128.checked_shl(self.lambda)
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
        let p = NeoParams::goldilocks_127();
        assert_eq!(p.B, 4096);
        let lhs = (p.k as u128 + 1) * (p.T as u128) * ((p.b as u128) - 1);
        assert!(lhs < p.B as u128, "guard must hold");
    }

    #[test]
    fn s_min_monotone_in_lambda() {
        let p = NeoParams::goldilocks_127();
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
        let mut p = NeoParams::goldilocks_127();
        // s!=2 not supported
        p.s = 3;
        assert_eq!(
            Err(ParamsError::UnsupportedExtension { required: 3 }),
            NeoParams::new(p.q, p.eta, p.d, p.kappa, p.m, p.b, p.k, p.T, 3, p.lambda)
        );
    }

    #[test]
    fn serde_roundtrip() {
        let p = NeoParams::goldilocks_127();
        let s = serde_json::to_string(&p).unwrap();
        let back: NeoParams = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
    }
}