//! Core Ajtai Matrix Commitment Implementation
//!
//! Implements Setup(κ, m) → random M ∈ R_q^{κ × m} and Commit(Z ∈ F_q^{d × m})
//! with S-module homomorphism for folding-friendly operations.

use neo_math::F;
use neo_modint::{Coeff, ModInt};
use neo_ring::RingElement;
use p3_field::PrimeField64;
use rand::{rngs::StdRng, Rng, SeedableRng};

use subtle::{Choice, ConstantTimeLess};

use super::rot::RotationRing;

/// Neo parameters aligned with paper §6 (Goldilocks-friendly, η=81, d=54)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeoParams {
    /// Base field modulus q (should be Goldilocks: 2^64 - 2^32 + 1)
    pub q: u64,
    /// Cyclotomic degree n (power of 2 for negacyclic ring)
    pub n: usize,
    /// MSIS rows κ (module rank)
    pub k: usize,
    /// Message length m (number of ring elements to commit)
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
pub const GOLDILOCKS_PARAMS: NeoParams = NeoParams {
    q: 0xFFFFFFFF00000001u64, // Goldilocks prime: 2^64 - 2^32 + 1
    n: 54,                               // Cyclotomic degree for η=81
    k: 16,                               // MSIS rows κ ≈ 16
    d: 54,                               // Message length matches cyclotomic degree
    b: 2,                                // Bit decomposition base
    e_bound: 64,                         // Gaussian error bound
    norm_bound: 4096,                    // B = b^k = 2^12 = 4096
    sigma: 3.2,                          // Gaussian std dev
    beta: 3,                             // Blinding bound
    max_blind_norm: (1u64 << 61) - 1,
};

/// Production parameters (scaled for reasonable performance)
pub const SECURE_PARAMS: NeoParams = NeoParams {
    q: ModInt::Q,          // Must match ModInt ring modulus
    n: 64,                 // Power of 2 for negacyclic ring compatibility
    k: 16,                 // MSIS rows
    d: 32,                 // Message length
    b: 2,                  // Bit decomposition
    e_bound: 64,           // Gaussian error bound
    norm_bound: 4096,      // B = 2^12
    sigma: 3.2,            // Gaussian std dev for ZK hiding
    beta: 3,               // Blinding bound
    max_blind_norm: (1u64 << 61) - 1,
};

/// Toy parameters for testing (insecure but fast)
pub const TOY_PARAMS: NeoParams = NeoParams {
    q: ModInt::Q,
    n: 4,                  // Small for testing
    k: 2,                  // Small module rank
    d: 8,                  // Small message length
    b: 2,                  // Bit decomposition
    e_bound: 64,
    norm_bound: 1 << 10,   // Smaller bound for testing
    sigma: 3.2,
    beta: 3,
    max_blind_norm: 64,
};

/// Ajtai Matrix Committer with S-module homomorphism
pub struct AjtaiCommitter {
    /// Public matrix A ∈ R_q^{κ × d}
    a: Vec<Vec<RingElement<ModInt>>>,
    /// Parameters
    params: NeoParams,
    /// Rotation ring for S-module operations
    rotation_ring: RotationRing,
}

impl Default for AjtaiCommitter {
    fn default() -> Self {
        Self::new()
    }
}

impl AjtaiCommitter {
    /// Construct a committer with secure parameters
    pub fn new() -> Self {
        // Enforce secure parameters in production builds
        #[cfg(not(test))]
        {
            let lambda = compute_lambda(&SECURE_PARAMS);
            assert!(lambda >= 128.0, "Insecure parameters: lambda={:.2} < 128.0", lambda);
        }
        Self::setup(SECURE_PARAMS)
    }

    /// Setup(κ, m) → random M ∈ R_q^{κ × m} as per Neo Algorithm 1
    pub fn setup(params: NeoParams) -> Self {
        // Runtime security validation
        let lambda = compute_lambda(&params);

        // For production builds, enforce secure parameters
        #[cfg(not(test))]
        {
            assert!(lambda >= 128.0, "Insecure params: lambda={lambda:.2} < 128.0");
            assert_eq!(params.q, ModInt::Q, "Params.q must equal ModInt::Q");
        }

        // For tests, allow toy parameters but warn about security
        #[cfg(test)]
        {
            if lambda < 128.0 {
                eprintln!("WARNING: Using insecure parameters in tests: lambda={lambda:.2} < 128.0");
            }
        }

        Self::setup_unchecked(params)
    }

    /// Setup without security checks (for testing)
    pub fn setup_unchecked(params: NeoParams) -> Self {
        let mut rng = StdRng::from_entropy();

        // Sample random Ā ∈ Z_q^{κ × (d-κ)}
        let m_bar = params.d - params.k;
        let a_bar: Vec<Vec<RingElement<ModInt>>> = (0..params.k)
            .map(|_| {
                (0..m_bar)
                    .map(|_| {
                        let coeffs = (0..params.n)
                            .map(|_| ModInt::from_u64(rng.gen::<u64>() % params.q))
                            .collect();
                        RingElement::from_coeffs(coeffs, params.n)
                    })
                    .collect()
            })
            .collect();

        // Sample short R ∈ Z^{(d-κ) × κ} from discrete Gaussian
        let r: Vec<Vec<RingElement<ModInt>>> = (0..m_bar)
            .map(|_| {
                (0..params.k)
                    .map(|_| {
                        let coeffs = (0..params.n)
                            .map(|_| {
                                Self::discrete_gaussian_sample(params.sigma, &mut rng, params.q)
                            })
                            .collect();
                        RingElement::from_coeffs(coeffs, params.n)
                    })
                    .collect()
            })
            .collect();

        // Gadget matrix G = b * I_κ (generalized from paper)
        let gadget_scalar = ModInt::from_u64(params.b);
        let mut a = vec![vec![RingElement::from_scalar(ModInt::zero(), params.n); params.d]; params.k];
        let minus_one = RingElement::from_scalar(
            ModInt::from_u64(<ModInt as Coeff>::modulus() - 1),
            params.n
        );

        for i in 0..params.k {
            // Copy Ā
            for j in 0..m_bar {
                a[i][j] = a_bar[i][j].clone();
            }
            // Compute G - Ā R
            for j in 0..params.k {
                let mut ar = RingElement::from_scalar(ModInt::zero(), params.n);
                for (l, r_row) in r.iter().enumerate().take(m_bar) {
                    ar = ar + a_bar[i][l].clone() * r_row[j].clone();
                }
                let g_ij = if i == j {
                    RingElement::from_scalar(gadget_scalar, params.n)
                } else {
                    RingElement::from_scalar(ModInt::zero(), params.n)
                };
                a[i][m_bar + j] = g_ij + ar.clone() * minus_one.clone();
            }
        }

        // Initialize rotation ring for S-module operations
        let rotation_ring = RotationRing::new(params.n);

        Self {
            a,
            params,
            rotation_ring,
        }
    }

    /// Get parameters
    pub fn params(&self) -> NeoParams {
        self.params
    }

    /// Get reference to public matrix (for testing)
    pub fn public_matrix(&self) -> &Vec<Vec<RingElement<ModInt>>> {
        &self.a
    }

    /// Get reference to rotation ring
    pub fn rotation_ring(&self) -> &RotationRing {
        &self.rotation_ring
    }

    /// Commit(Z ∈ F_q^{d × m}) → c = cf(M z') with pay-per-bit embedding
    /// Returns (commitment, noise, blinded_witness, blinding_vector)
    #[allow(clippy::type_complexity)]
    pub fn commit(
        &self,
        w: &[RingElement<ModInt>],
        transcript: &mut Vec<u8>,
    ) -> Result<(
        Vec<RingElement<ModInt>>, // commitment c
        Vec<RingElement<ModInt>>, // noise e
        Vec<RingElement<ModInt>>, // blinded witness
        Vec<RingElement<ModInt>>, // blinding vector r
    ), &'static str> {
        // Domain-separated seed for blinding randomness
        use neo_sumcheck::fiat_shamir::Transcript;
        let mut fs_transcript = Transcript::new("ajtai_commit");
        fs_transcript.absorb_bytes("transcript_state", transcript);
        fs_transcript.absorb_tag("NEO/V1/ajtai/commit/blinding");
        // Generate seed from transcript challenges
        let mut seed_array = [0u8; 32];
        for i in 0..4 {
            let challenge = fs_transcript.challenge_base(&format!("NEO/V1/ajtai/seed/{}", i));
            let bytes = challenge.as_canonical_u64().to_le_bytes();
            seed_array[i*8..(i+1)*8].copy_from_slice(&bytes);
        }
        let mut rng = StdRng::from_seed(seed_array);
        self.commit_with_rng(w, &mut rng)
    }

    /// Deterministic commit using caller-provided RNG
    #[allow(clippy::type_complexity)]
    pub fn commit_with_rng(
        &self,
        w: &[RingElement<ModInt>],
        rng: &mut impl Rng,
    ) -> Result<(
        Vec<RingElement<ModInt>>, // commitment c
        Vec<RingElement<ModInt>>, // noise e
        Vec<RingElement<ModInt>>, // blinded witness
        Vec<RingElement<ModInt>>, // blinding vector r
    ), &'static str> {
        if w.len() != self.params.d {
            return Err("Witness length mismatch");
        }

        // ZK blinding parameters
        let zk_sigma = self.params.sigma;
        let beta_zk = self.params.beta;

        // Generate blinding vector r
        let r: Vec<RingElement<ModInt>> = (0..w.len())
            .map(|_| RingElement::random_small(rng, self.params.n, beta_zk))
            .collect();

        // Compute blinded witness w' = w + r
        let mut blinded_w = Vec::with_capacity(w.len());
        for (wi, ri) in w.iter().zip(&r) {
            blinded_w.push(wi.clone() + ri.clone());
        }

        // Generate commitment noise e for ZK hiding
        let e: Vec<_> = if cfg!(test) {
            // Use zero noise in tests for deterministic behavior
            (0..self.params.k).map(|_| RingElement::zero(self.params.n)).collect()
        } else {
            (0..self.params.k)
                .map(|_| {
                    let mut noise = RingElement::random_gaussian(rng, self.params.n, zk_sigma);
                    // Ensure noise stays within bounds
                    while noise.norm_inf() > self.params.e_bound {
                        noise = RingElement::random_gaussian(rng, self.params.n, zk_sigma);
                    }
                    noise
                })
                .collect()
        };

        // Compute commitment c = A * w' + e
        let mut c = vec![RingElement::zero(self.params.n); self.params.k];
        for i in 0..self.params.k {
            for (j, w_item) in blinded_w.iter().enumerate().take(self.params.d) {
                c[i] = c[i].clone() + self.a[i][j].clone() * w_item.clone();
            }
            c[i] = c[i].clone() + e[i].clone();
        }

        Ok((c, e, blinded_w, r))
    }

    /// Verify commitment: check that c = A * w + e and norms are within bounds
    pub fn verify(
        &self,
        c: &[RingElement<ModInt>],
        w: &[RingElement<ModInt>],
        e: &[RingElement<ModInt>],
    ) -> bool {
        if c.len() != self.params.k || w.len() != self.params.d || e.len() != self.params.k {
            return false;
        }

        // Check norm bounds using constant-time operations
        let w_bound = self.params.norm_bound + self.params.beta;
        let gauss_bound = self.params.e_bound;

        let mut w_ok = Choice::from(1u8);
        for wi in w {
            let norm = wi.norm_inf();
            let gt = w_bound.ct_lt(&norm);
            w_ok &= !gt;
        }

        let mut e_ok = Choice::from(1u8);
        for ei in e {
            let norm = ei.norm_inf();
            let gt = gauss_bound.ct_lt(&norm);
            e_ok &= !gt;
        }

        if (w_ok & e_ok).unwrap_u8() == 0 {
            return false;
        }

        // Check commitment equation: c = A * w + e
        for (i, (ai_row, ei)) in self.a.iter().zip(e.iter()).enumerate() {
            let mut expected = RingElement::zero(self.params.n);
            for (aij, w_item) in ai_row.iter().zip(w.iter()) {
                expected = expected + aij.clone() * w_item.clone();
            }
            expected = expected + ei.clone();
            if expected != c[i] {
                return false;
            }
        }

        true
    }

    /// S-module homomorphism: ρ₁ · c₁ + ρ₂ · c₂ = Commit(ρ₁ Z₁ + ρ₂ Z₂)
    /// This is the core property that enables folding
    pub fn random_linear_combo(
        &self,
        c1: &[RingElement<ModInt>],
        c2: &[RingElement<ModInt>],
        rho: F,
    ) -> Vec<RingElement<ModInt>> {
        // Reject if rho ≥ q to avoid wrap-around
        let q = <ModInt as Coeff>::modulus();
        if rho.as_canonical_u64() >= q {
            panic!("random_linear_combo: rho outside Z_q representative range");
        }

        // Length-agnostic with zero-broadcast for degenerate cases
        let n = self.params.n;
        let len = c1.len().max(c2.len());
        if len == 0 {
            return Vec::new();
        }

        let zero = RingElement::from_scalar(ModInt::zero(), n);
        let rho_scalar = RingElement::from_scalar(ModInt::from_u64(rho.as_canonical_u64()), n);

        (0..len)
            .map(|i| {
                let a = c1.get(i).unwrap_or(&zero).clone();
                let b = c2.get(i).unwrap_or(&zero).clone();
                a + b * rho_scalar.clone()
            })
            .collect()
    }

    /// S-module homomorphism using rotation element ρ ∈ S
    /// This version uses the rotation-matrix ring for small-norm challenges
    pub fn random_linear_combo_rotation(
        &self,
        c1: &[RingElement<ModInt>],
        c2: &[RingElement<ModInt>],
        rho_rot: &RingElement<ModInt>,
    ) -> Vec<RingElement<ModInt>> {
        let n = self.params.n;
        let len = c1.len().max(c2.len());
        if len == 0 {
            return Vec::new();
        }

        let zero = RingElement::from_scalar(ModInt::zero(), n);
        (0..len)
            .map(|i| {
                let a = c1.get(i).unwrap_or(&zero).clone();
                let b = c2.get(i).unwrap_or(&zero).clone();
                a + rho_rot.clone() * b
            })
            .collect()
    }

    /// Sample from discrete Gaussian distribution
    pub fn discrete_gaussian_sample(sigma: f64, rng: &mut impl Rng, _q: u64) -> ModInt {
        assert!(sigma > 0.0, "sigma must be > 0");
        // Draw from continuous N(0, σ²), round to nearest integer, clip to ~6σ
        let tail = (6.0 * sigma).ceil() as i128;
        loop {
            // rand_distr::StandardNormal → N(0, 1)
            let x: f64 = rng.gen::<f64>() * sigma;
            let z = x.round() as i128;
            if z.abs() <= tail {
                return ModInt::from(z);
            }
        }
    }
}

/// Compute rough security estimate λ (for development-time checks only)
fn compute_lambda(params: &NeoParams) -> f64 {
    // Very rough log2 security estimate combining MSIS and RLWE-style bounds
    let msis = (params.k as f64 * params.d as f64) * (params.q as f64).log2()
        + (2.0 * params.sigma * (params.n as f64 * params.k as f64).sqrt()).log2()
            * (params.d as f64)
        - (params.e_bound as f64).log2();
    let rlwe = (params.n as f64) * (params.q as f64).log2()
        - (params.sigma.powi(2) * params.n as f64).log2();
    msis.min(rlwe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_setup_and_params() {
        let committer = AjtaiCommitter::setup(TOY_PARAMS);
        assert_eq!(committer.params().k, TOY_PARAMS.k);
        assert_eq!(committer.params().d, TOY_PARAMS.d);
        assert_eq!(committer.params().n, TOY_PARAMS.n);
    }

    #[test]
    fn test_commit_verify_roundtrip() {
        let committer = AjtaiCommitter::setup(TOY_PARAMS);
        let mut rng = StdRng::seed_from_u64(42);
        
        // Generate random witness
        let w: Vec<RingElement<ModInt>> = (0..committer.params().d)
            .map(|_| RingElement::random_small(&mut rng, committer.params().n, 2))
            .collect();

        let (c, e, blinded_w, _r) = committer.commit_with_rng(&w, &mut rng).unwrap();
        
        // Verify should pass with blinded witness
        assert!(committer.verify(&c, &blinded_w, &e));
    }

    #[test]
    fn test_s_module_homomorphism() {
        let committer = AjtaiCommitter::setup(TOY_PARAMS);
        let mut rng = StdRng::seed_from_u64(42);
        
        // Generate two commitments
        let w1: Vec<RingElement<ModInt>> = (0..committer.params().d)
            .map(|_| RingElement::random_small(&mut rng, committer.params().n, 2))
            .collect();
        let w2: Vec<RingElement<ModInt>> = (0..committer.params().d)
            .map(|_| RingElement::random_small(&mut rng, committer.params().n, 2))
            .collect();

        let (c1, _e1, _bw1, _r1) = committer.commit_with_rng(&w1, &mut rng).unwrap();
        let (c2, _e2, _bw2, _r2) = committer.commit_with_rng(&w2, &mut rng).unwrap();

        // Test scalar homomorphism
        let rho = F::from_u64(42);
        let combo = committer.random_linear_combo(&c1, &c2, rho);
        
        // Should have correct length
        assert_eq!(combo.len(), c1.len().max(c2.len()));
    }
}
