//! Parameter sets for Neo protocol
//!
//! This module defines the core parameter sets used throughout Neo,
//! aligned with the security analysis in §6 of the paper.

/// Core Neo parameters for a specific instantiation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeoParams {
    /// Base field modulus q
    pub q: u64,
    /// Cyclotomic degree n (power of 2 for negacyclic rings)
    pub n: usize,
    /// MSIS module rank κ
    pub k: usize,
    /// Message length d (number of ring elements to commit)
    pub d: usize,
    /// Decomposition base b (typically 2 for bit decomposition)
    pub b: u64,
    /// Gaussian error bound for commitment noise
    pub e_bound: u64,
    /// Norm bound for witness vectors
    pub norm_bound: u64,
    /// Gaussian standard deviation for sampling
    pub sigma: f64,
    /// Blinding bound for zero-knowledge
    pub beta: u64,
    /// Maximum norm for blinded values
    pub max_blind_norm: u64,
}

/// Goldilocks parameters (η=81, d=54) as specified in Neo §6.2
/// 
/// These parameters provide ~128-bit security with efficient arithmetic
/// over the Goldilocks field (2^64 - 2^32 + 1).
pub const GOLDILOCKS_PARAMS: NeoParams = NeoParams {
    q: 0xFFFFFFFF00000001u64, // Goldilocks prime: 2^64 - 2^32 + 1
    n: 54,                    // Cyclotomic degree for η=81
    k: 16,                    // MSIS rows κ ≈ 16
    d: 54,                    // Message length matches cyclotomic degree
    b: 2,                     // Bit decomposition base
    e_bound: 64,              // Gaussian error bound
    norm_bound: 4096,         // B = b^k = 2^12 = 4096
    sigma: 3.2,               // Gaussian std dev
    beta: 3,                  // Blinding bound
    max_blind_norm: (1u64 << 61) - 1,
};

/// Mersenne-61 parameters for alternative field choice
/// 
/// Uses the Mersenne prime 2^61 - 1 for efficient modular arithmetic
/// with similar security properties to Goldilocks.
pub const MERSENNE61_PARAMS: NeoParams = NeoParams {
    q: (1u64 << 61) - 1,     // Mersenne prime 2^61 - 1
    n: 64,                    // Power of 2 for negacyclic ring compatibility
    k: 16,                    // MSIS rows
    d: 32,                    // Message length
    b: 2,                     // Bit decomposition
    e_bound: 64,              // Gaussian error bound
    norm_bound: 4096,         // B = 2^12
    sigma: 3.2,               // Gaussian std dev for ZK hiding
    beta: 3,                  // Blinding bound
    max_blind_norm: (1u64 << 61) - 1,
};

/// Toy parameters for testing and development
/// 
/// **WARNING**: These parameters are insecure and should only be used for testing.
pub const TOY_PARAMS: NeoParams = NeoParams {
    q: (1u64 << 61) - 1,     // Use same field as Mersenne for simplicity
    n: 4,                     // Small for testing
    k: 2,                     // Small module rank
    d: 8,                     // Small message length
    b: 2,                     // Bit decomposition
    e_bound: 64,
    norm_bound: 1 << 10,      // Smaller bound for testing
    sigma: 3.2,
    beta: 3,
    max_blind_norm: 64,
};

impl NeoParams {
    /// Compute rough security estimate λ (for development-time checks only)
    /// 
    /// This provides a very rough log2 security estimate combining MSIS and RLWE-style bounds.
    /// For production use, run the full security analysis from Appendix B.10.
    pub fn security_estimate(&self) -> f64 {
        let msis = (self.k as f64 * self.d as f64) * (self.q as f64).log2()
            + (2.0 * self.sigma * (self.n as f64 * self.k as f64).sqrt()).log2()
                * (self.d as f64)
            - (self.e_bound as f64).log2();
        let rlwe = (self.n as f64) * (self.q as f64).log2()
            - (self.sigma.powi(2) * self.n as f64).log2();
        msis.min(rlwe)
    }

    /// Check if parameters meet minimum security requirements
    pub fn is_secure(&self) -> bool {
        self.security_estimate() >= 128.0
    }

    /// Validate parameter consistency
    pub fn validate(&self) -> Result<(), String> {
        if !self.n.is_power_of_two() {
            return Err("Cyclotomic degree n must be a power of 2".to_string());
        }
        if self.d < self.k {
            return Err("Message length d must be >= module rank k".to_string());
        }
        if self.b < 2 {
            return Err("Decomposition base b must be >= 2".to_string());
        }
        if self.sigma <= 0.0 {
            return Err("Gaussian standard deviation sigma must be > 0".to_string());
        }
        Ok(())
    }
}

/// Convenience constructors for common parameter sets
impl NeoParams {
    /// Goldilocks parameters with 128-bit security
    pub fn goldilocks() -> Self {
        GOLDILOCKS_PARAMS
    }

    /// Mersenne-61 parameters with 128-bit security  
    pub fn mersenne61() -> Self {
        MERSENNE61_PARAMS
    }

    /// Toy parameters for testing (insecure)
    pub fn toy() -> Self {
        TOY_PARAMS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_validation() {
        assert!(GOLDILOCKS_PARAMS.validate().is_ok());
        assert!(MERSENNE61_PARAMS.validate().is_ok());
        assert!(TOY_PARAMS.validate().is_ok());
    }

    #[test]
    fn test_security_estimates() {
        assert!(GOLDILOCKS_PARAMS.is_secure());
        assert!(MERSENNE61_PARAMS.is_secure());
        // Toy parameters are intentionally insecure
        assert!(!TOY_PARAMS.is_secure());
    }

    #[test]
    fn test_convenience_constructors() {
        assert_eq!(NeoParams::goldilocks(), GOLDILOCKS_PARAMS);
        assert_eq!(NeoParams::mersenne61(), MERSENNE61_PARAMS);
        assert_eq!(NeoParams::toy(), TOY_PARAMS);
    }
}
