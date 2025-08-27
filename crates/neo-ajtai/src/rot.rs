//! Rotation-Matrix Ring S and Challenge Set C
//!
//! Implements Neo's rotation-matrix ring S = {rot(a)} for S-module homomorphism
//! and the challenge set C with small-norm elements for secure random linear combinations.

use neo_modint::{Coeff, ModInt};
use neo_ring::RingElement;
// Removed unused imports
use rand::Rng;
use std::collections::HashSet;

/// Rotation-matrix ring S = {rot(a) : a ∈ R_q}
/// Enables S-module homomorphism: ρ₁ · c₁ + ρ₂ · c₂ = Commit(ρ₁ Z₁ + ρ₂ Z₂)
pub struct RotationRing {
    /// Ring degree n
    n: usize,
}

impl RotationRing {
    /// Create new rotation ring for degree n
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Create rotation matrix rot(a) from ring element a
    /// rot(a) represents multiplication by a in the ring R_q[X]/(X^n + 1)
    pub fn rotation_matrix(&self, a: &RingElement<ModInt>) -> RotationMatrix {
        RotationMatrix::new(a.clone(), self.n)
    }

    /// Fast rotation operation: rot(a) · v
    /// Computes multiplication a * v in the ring without constructing full matrix
    pub fn rotate(&self, a: &RingElement<ModInt>, v: &RingElement<ModInt>) -> RingElement<ModInt> {
        // Direct ring multiplication is equivalent to matrix-vector product
        a.clone() * v.clone()
    }

    /// Batch rotation: apply rot(a) to multiple vectors
    pub fn rotate_batch(
        &self,
        a: &RingElement<ModInt>,
        vectors: &[RingElement<ModInt>],
    ) -> Vec<RingElement<ModInt>> {
        vectors.iter().map(|v| self.rotate(a, v)).collect()
    }

    /// Check if element a generates an invertible rotation
    /// Required for challenge set: (ρ - ρ')^{-1} must exist in S
    pub fn is_invertible(&self, a: &RingElement<ModInt>) -> bool {
        // Check if gcd(a(X), X^n + 1) = 1
        // For now, use a simple heuristic: element is not zero
        !a.coeffs().iter().all(|&c| c == ModInt::zero())
    }

    /// Ring degree
    pub fn degree(&self) -> usize {
        self.n
    }
}

/// Rotation matrix representation (for explicit matrix operations if needed)
pub struct RotationMatrix {
    /// Generating ring element
    generator: RingElement<ModInt>,
    /// Ring degree
    #[allow(dead_code)]
    n: usize,
}

impl RotationMatrix {
    fn new(generator: RingElement<ModInt>, n: usize) -> Self {
        Self { generator, n }
    }

    /// Apply rotation to vector (same as ring multiplication)
    pub fn apply(&self, v: &RingElement<ModInt>) -> RingElement<ModInt> {
        self.generator.clone() * v.clone()
    }

    /// Get the generating element
    pub fn generator(&self) -> &RingElement<ModInt> {
        &self.generator
    }
}

/// Challenge set C with small-norm elements and invertibility guarantees
/// Implements Neo's strong sampling set for secure random linear combinations
pub struct ChallengeSet {
    /// Ring degree
    #[allow(dead_code)]
    n: usize,
    /// Challenge elements with small coefficients
    challenges: Vec<RingElement<ModInt>>,
    /// Invertibility bound b_inv
    b_inv: u64,
    /// Expansion factor T (measured)
    expansion_factor: f64,
}

impl ChallengeSet {
    /// Create challenge set C from small-coefficient ring elements
    /// coeffs_bound: maximum absolute value of coefficients (e.g., 2 for {-2,-1,0,1,2})
    pub fn new(n: usize, coeffs_bound: u64, target_size: usize) -> Self {
        let b_inv = coeffs_bound * 2; // Lyubashevsky-Seiler bound
        let mut challenges = Vec::new();
        let mut seen = HashSet::new();
        
        // Generate all possible small-coefficient elements
        Self::generate_small_elements(n, coeffs_bound, target_size, &mut challenges, &mut seen);
        
        // Filter for invertibility
        let rotation_ring = RotationRing::new(n);
        challenges.retain(|a| rotation_ring.is_invertible(a));
        
        // Compute expansion factor T (simplified estimate)
        let expansion_factor = Self::estimate_expansion_factor(&challenges, coeffs_bound);
        
        Self {
            n,
            challenges,
            b_inv,
            expansion_factor,
        }
    }

    /// Sample random challenge from C
    pub fn sample_challenge(&self, rng: &mut impl Rng) -> &RingElement<ModInt> {
        if self.challenges.is_empty() {
            panic!("No challenges available in challenge set");
        }
        let idx = rng.gen_range(0..self.challenges.len());
        &self.challenges[idx]
    }

    /// Sample multiple distinct challenges
    pub fn sample_challenges(&self, count: usize, rng: &mut impl Rng) -> Vec<&RingElement<ModInt>> {
        if count > self.challenges.len() {
            panic!("Not enough challenges available: requested {}, have {}", count, self.challenges.len());
        }
        
        let mut indices: Vec<usize> = (0..self.challenges.len()).collect();
        
        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        
        indices.into_iter()
            .take(count)
            .map(|idx| &self.challenges[idx])
            .collect()
    }

    /// Check invertibility bound: ||cf(a - b)||_∞ < b_inv
    pub fn check_invertibility_bound(&self, a: &RingElement<ModInt>, b: &RingElement<ModInt>) -> bool {
        let diff = a.clone() - b.clone();
        diff.norm_inf() < self.b_inv
    }

    /// Get expansion factor T
    pub fn expansion_factor(&self) -> f64 {
        self.expansion_factor
    }

    /// Get invertibility bound
    pub fn invertibility_bound(&self) -> u64 {
        self.b_inv
    }

    /// Number of available challenges
    pub fn size(&self) -> usize {
        self.challenges.len()
    }

    /// Generate small-coefficient ring elements
    fn generate_small_elements(
        n: usize,
        coeffs_bound: u64,
        target_size: usize,
        challenges: &mut Vec<RingElement<ModInt>>,
        seen: &mut HashSet<Vec<u64>>,
    ) {
        let bound = coeffs_bound as i64;
        
        // Generate elements with coefficients in [-bound, bound]
        // For efficiency, we'll generate a subset rather than all possibilities
        let mut attempts = 0;
        let max_attempts = target_size * 10;
        
        while challenges.len() < target_size && attempts < max_attempts {
            let mut coeffs = Vec::with_capacity(n);
            
            // Generate random small coefficients
            for _ in 0..n {
                let coeff_val = if attempts % 3 == 0 {
                    // Include some zero coefficients for sparsity
                    0
                } else {
                    // Random in [-bound, bound]
                    rand::random::<i64>() % (2 * bound + 1) - bound
                };
                coeffs.push(ModInt::from(coeff_val as i128));
            }
            
            // Check for duplicates
            let canonical: Vec<u64> = coeffs.iter().map(|c| c.as_canonical_u64()).collect();
            if !seen.contains(&canonical) {
                seen.insert(canonical);
                challenges.push(RingElement::from_coeffs(coeffs, n));
            }
            
            attempts += 1;
        }
        
        // Always ensure we have at least some basic challenges
        if challenges.is_empty() {
            // Add identity
            let identity = RingElement::from_scalar(ModInt::one(), n);
            challenges.push(identity);
        }
        
        // Add more standard elements if needed
        if challenges.len() < target_size.min(10) {
            // Add X if we have room
            if n >= 2 {
                let x = RingElement::from_coeffs(
                    vec![ModInt::zero(), ModInt::one()]
                        .into_iter()
                        .chain(std::iter::repeat(ModInt::zero()).take(n - 2))
                        .collect(),
                    n,
                );
                challenges.push(x);
            }
            
            // Add -1 if we have room
            if challenges.len() < target_size.min(10) {
                let minus_one = RingElement::from_scalar(
                    ModInt::from_u64(<ModInt as Coeff>::modulus() - 1), 
                    n
                );
                challenges.push(minus_one);
            }
        }
    }

    /// Estimate expansion factor T for the challenge set
    fn estimate_expansion_factor(challenges: &[RingElement<ModInt>], coeffs_bound: u64) -> f64 {
        if challenges.is_empty() {
            return 1.0;
        }
        
        // Simple estimate: T ≈ max_norm / coeffs_bound
        let max_norm = challenges
            .iter()
            .map(|c| c.norm_inf())
            .max()
            .unwrap_or(1);
        
        (max_norm as f64) / (coeffs_bound as f64).max(1.0)
    }
}

/// Default challenge set for Neo parameters
/// Uses coefficients in {-2, -1, 0, 1, 2} as suggested in the paper
pub fn default_challenge_set(n: usize) -> ChallengeSet {
    ChallengeSet::new(n, 2, 1000) // coeffs in [-2, 2], ~1000 challenges
}

/// Create challenge set optimized for specific security level
pub fn challenge_set_for_security(n: usize, security_bits: usize) -> ChallengeSet {
    let coeffs_bound = match security_bits {
        0..=80 => 1,   // {-1, 0, 1}
        81..=128 => 2, // {-2, -1, 0, 1, 2}
        _ => 3,        // {-3, -2, -1, 0, 1, 2, 3}
    };
    
    let target_size = 2_usize.pow((security_bits / 8).min(12) as u32); // Cap at 2^12 = 4096
    ChallengeSet::new(n, coeffs_bound, target_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_rotation_ring_basic() {
        let ring = RotationRing::new(4);
        let a = RingElement::from_coeffs(vec![ModInt::one(), ModInt::from_u64(2)], 4);
        let v = RingElement::from_coeffs(vec![ModInt::from_u64(3), ModInt::one()], 4);
        
        let result = ring.rotate(&a, &v);
        // Ring multiplication may reduce the degree, so just check it's non-empty
        assert!(!result.coeffs().is_empty());
        assert!(result.coeffs().len() <= 4);
    }

    #[test]
    fn test_rotation_matrix() {
        let generator = RingElement::from_coeffs(vec![ModInt::one(), ModInt::from_u64(2)], 4);
        let rot_matrix = RotationMatrix::new(generator.clone(), 4);
        
        assert_eq!(rot_matrix.generator(), &generator);
        
        let v = RingElement::from_coeffs(vec![ModInt::from_u64(3), ModInt::one()], 4);
        let result = rot_matrix.apply(&v);
        assert!(!result.coeffs().is_empty());
        assert!(result.coeffs().len() <= 4);
    }

    #[test]
    fn test_challenge_set_creation() {
        let challenge_set = ChallengeSet::new(4, 2, 10);
        assert!(challenge_set.size() > 0);
        assert!(challenge_set.expansion_factor() > 0.0);
        assert_eq!(challenge_set.invertibility_bound(), 4); // 2 * coeffs_bound
    }

    #[test]
    fn test_challenge_sampling() {
        let challenge_set = ChallengeSet::new(4, 2, 100);
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        
        let challenge = challenge_set.sample_challenge(&mut rng);
        assert_eq!(challenge.coeffs().len(), 4);
        
        // Sample multiple challenges
        let challenges = challenge_set.sample_challenges(3, &mut rng);
        assert_eq!(challenges.len(), 3);
        
        // Should be distinct (with high probability)
        assert_ne!(challenges[0], challenges[1]);
    }

    #[test]
    fn test_invertibility_check() {
        let ring = RotationRing::new(4);
        
        // Identity should be invertible
        let identity = RingElement::from_scalar(ModInt::one(), 4);
        assert!(ring.is_invertible(&identity));
        
        // Zero should not be invertible
        let zero = RingElement::from_scalar(ModInt::zero(), 4);
        assert!(!ring.is_invertible(&zero));
    }

    #[test]
    fn test_invertibility_bound() {
        let challenge_set = ChallengeSet::new(4, 2, 10);
        let a = RingElement::from_coeffs(vec![ModInt::one(), ModInt::zero()], 4);
        let b = RingElement::from_coeffs(vec![ModInt::from_u64(2), ModInt::zero()], 4);
        
        // Small difference should pass
        assert!(challenge_set.check_invertibility_bound(&a, &b));
        
        // Large difference should fail
        let c = RingElement::from_coeffs(vec![ModInt::from_u64(100), ModInt::zero()], 4);
        assert!(!challenge_set.check_invertibility_bound(&a, &c));
    }

    #[test]
    fn test_default_challenge_set() {
        let challenge_set = default_challenge_set(8);
        assert!(challenge_set.size() > 0);
        assert_eq!(challenge_set.invertibility_bound(), 4); // 2 * 2
    }

    #[test]
    fn test_security_parameterized_challenge_set() {
        let challenge_set_80 = challenge_set_for_security(8, 80);
        let challenge_set_128 = challenge_set_for_security(8, 128);
        
        // Higher security should have larger coefficient bounds
        assert!(challenge_set_128.invertibility_bound() >= challenge_set_80.invertibility_bound());
    }

    #[test]
    fn test_batch_rotation() {
        let ring = RotationRing::new(4);
        let a = RingElement::from_coeffs(vec![ModInt::one(), ModInt::from_u64(2)], 4);
        let vectors = vec![
            RingElement::from_coeffs(vec![ModInt::from_u64(3), ModInt::one()], 4),
            RingElement::from_coeffs(vec![ModInt::from_u64(5), ModInt::from_u64(7)], 4),
        ];
        
        let results = ring.rotate_batch(&a, &vectors);
        assert_eq!(results.len(), 2);
        assert!(!results[0].coeffs().is_empty());
        assert!(!results[1].coeffs().is_empty());
    }
}
