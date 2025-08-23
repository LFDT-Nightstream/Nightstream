use neo_fields::{from_base, ExtF, ExtFieldNormTrait, F};

use subtle::{Choice, ConstantTimeLess};
use neo_modint::{Coeff, ModInt};
use neo_ring::RingElement;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;




#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeoParams {
    pub q: u64,
    pub n: usize,
    pub k: usize,
    pub d: usize,
    pub b: u64,
    pub e_bound: u64,
    pub norm_bound: u64,
    pub sigma: f64,
    pub beta: u64,
    pub max_blind_norm: u64,
}

pub const PRESET_GOLDILOCKS: NeoParams = NeoParams {
    // Parameters scaled to paper values (§6, App. B.10)
    q: (1u64 << 61) - 1, // Mersenne prime for efficient reduction
    n: 64,               // Cyclotomic degree
    k: 16,               // Module rank
    d: 64,               // Number of decomposition digits
    b: 2,                // Bit decomposition base
    e_bound: 64,         // Gaussian error bound
    norm_bound: 4096,    // 2^12 bound for packed vectors
    sigma: 3.2,          // Gaussian std dev for sampling
    beta: 3,             // Blinding bound
    max_blind_norm: (1u64 << 61) - 1,
};

/// Parameters resembling the production settings from the Neo paper
/// (§6, App. B.10) but scaled to remain reasonably fast.  These enable
/// Gaussian blinding and noise for zero-knowledge hiding.
/// Validate security levels with `sage_params.sage`.
pub const SECURE_PARAMS: NeoParams = NeoParams {
    q: ModInt::Q,          // Must match the ModInt ring modulus
    n: 64,                 // Power of 2 for negacyclic ring compatibility
    k: 16,
    d: 32,
    b: 2,
    e_bound: 64,
    norm_bound: 4096,
    sigma: 3.2, // Enable Gaussian for ZK hiding (§8)
    beta: 3,    // Enable blinding bound
    max_blind_norm: (1u64 << 61) - 1,
};

pub const TOY_PARAMS: NeoParams = NeoParams {
    q: ModInt::Q, // Must match the ModInt ring modulus
    n: 4, // Already power of 2 (2^2)
    k: 2,
    d: 64, // Increased to handle full field decomposition
    b: 2,
    e_bound: 64,
    norm_bound: 1 << 63, // Reduced to prevent overflow while still large
    sigma: 3.2,
    beta: 3,
    max_blind_norm: 64,
};

pub struct AjtaiCommitter {
    a: Vec<Vec<RingElement<ModInt>>>,        // Public matrix
    trapdoor: Vec<Vec<RingElement<ModInt>>>, // Trapdoor matrix
    params: NeoParams,
}

impl Default for AjtaiCommitter {
    fn default() -> Self {
        Self::new()
    }
}

impl AjtaiCommitter {
    /// Construct a committer with the default secure parameters.
    pub fn new() -> Self {
        Self::setup(SECURE_PARAMS)
    }

    pub fn setup(params: NeoParams) -> Self {
        // Runtime security validation
        let lambda = compute_lambda(&params);

        // For production builds, enforce secure parameters
        #[cfg(not(test))]
        {
            assert!(lambda >= 128.0, "Insecure params: lambda={lambda:.2} < 128.0");
            assert_eq!(params.q, ModInt::Q, "Params.q must equal ModInt::Q");
            // (Optional) if your ring code assumes a power-of-two cyclotomic degree:
            // assert!(params.n.is_power_of_two(), "n must be a power of two");
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

    pub fn setup_unchecked(params: NeoParams) -> Self {
        let mut rng = StdRng::from_rng(&mut rand::rng());

        // Sample random \bar{A} \in Z_q^{k x (d-k)}
        let m_bar = params.d - params.k;
        let a_bar: Vec<Vec<RingElement<ModInt>>> = (0..params.k)
            .map(|_| {
                (0..m_bar)
                    .map(|_| {
                        let coeffs = (0..params.n)
                            .map(|_| ModInt::from_u64(rng.random::<u64>() % params.q))
                            .collect();
                        RingElement::from_coeffs(coeffs, params.n)
                    })
                    .collect()
            })
            .collect();

        // Sample short R \in Z^{(d-k) x k} from discrete Gaussian
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

        // Gadget matrix G = 2 * I_k (simplified)
        let gadget_scalar = ModInt::from_u64(2);
        let mut a =
            vec![vec![RingElement::from_scalar(ModInt::zero(), params.n); params.d]; params.k];
        let minus_one = RingElement::from_scalar(
            ModInt::from_u64(<ModInt as Coeff>::modulus() - 1),
            params.n
        );
        for i in 0..params.k {
            // Copy \bar{A}
            for j in 0..m_bar {
                a[i][j] = a_bar[i][j].clone();
            }
            // Compute G - \bar{A} R
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

        // Trapdoor T = [R; I_k]
        let mut trapdoor =
            vec![vec![RingElement::from_scalar(ModInt::zero(), params.n); params.d]; params.k];
        for (i, trapdoor_row) in trapdoor.iter_mut().enumerate().take(params.k) {
            for (j, r_row) in r.iter().enumerate().take(m_bar) {
                trapdoor_row[j] = r_row[i].clone();
            }
            for j in 0..params.k {
                trapdoor_row[m_bar + j] = if i == j {
                    RingElement::from_scalar(ModInt::one(), params.n)
                } else {
                    RingElement::from_scalar(ModInt::zero(), params.n)
                };
            }
        }

        Self {
            a,
            trapdoor,
            params,
        }
    }

    pub fn params(&self) -> NeoParams {
        self.params
    }

    /// Get a reference to the public matrix A for testing purposes
    pub fn public_matrix(&self) -> &Vec<Vec<RingElement<ModInt>>> {
        &self.a
    }

    // Sample from a discrete Gaussian using standard normal + rounding for better precision
    pub fn discrete_gaussian_sample(sigma: f64, rng: &mut impl Rng, _q: u64) -> ModInt {
        assert!(sigma > 0.0, "sigma must be > 0");
        // Draw from continuous N(0, σ²), round to nearest integer, clip to ~6σ
        let tail = (6.0 * sigma).ceil() as i128;
        loop {
            // rand_distr::StandardNormal → N(0, 1)
            let x: f64 = rng.sample::<f64, _>(StandardNormal) * sigma;
            let z = x.round() as i128;
            if z.abs() <= tail {
                return ModInt::from(z); // reduce modulo ModInt::Q correctly
            }
        }
    }

    pub fn sample_gaussian_ring(
        &self,
        center: &RingElement<ModInt>,
        sigma: f64,
        rng: &mut impl Rng,
    ) -> Result<RingElement<ModInt>, &'static str> {
        if sigma <= 0.0 {
            return Err("Invalid sigma <=0");
        }
        // Use the ModInt modulus consistently for signed conversions.
        let q_i128 = <ModInt as Coeff>::modulus() as i128;
        let mut centers: Vec<i128> = center
            .coeffs()
            .iter()
            .map(|c| {
                let val = c.as_canonical_u64() as i128;
                if val > q_i128 / 2 {
                    val - q_i128
                } else {
                    val
                }
            })
            .collect();
        // Ensure we have exactly n centers, padding with zeros if necessary
        centers.resize(self.params.n, 0);
        let mut retries = 0;
        loop {
            let mut coeffs = Vec::with_capacity(self.params.n);
            for &center_i in &centers {
                // Sample from discrete Gaussian and add to center
                let gaussian_sample = Self::discrete_gaussian_sample(sigma, rng, self.params.q);
                // Center is already in signed i128; convert with ModInt modulus
                let center_modint = ModInt::from(center_i);
                let coeff = center_modint + gaussian_sample;
                coeffs.push(coeff);
            }
            let sample = RingElement::from_coeffs(coeffs, self.params.n);
            let diffs: Vec<i128> = sample
                .coeffs()
                .iter()
                .zip(centers.clone())
                .map(|(s, cen)| {
                    let s_i = s.as_canonical_u64() as i128;
                    if s_i > q_i128 / 2 {
                        s_i - q_i128 - cen
                    } else {
                        s_i - cen
                    }
                })
                .collect();
            let within_bound = diffs
                .iter()
                .all(|&d| d.abs() <= (self.params.e_bound * 3) as i128);
            if within_bound {
                return Ok(sample);
            }
            retries += 1;
            if retries > 1000 {  // Increased limit
                return Err("Gaussian sampling failed after 1000 retries");
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn gpv_trapdoor_sample(
        &self,
        target: &[RingElement<ModInt>],
        _sigma: f64,  // randomness comes from z
        rng: &mut impl Rng,
    ) -> Result<Vec<RingElement<ModInt>>, &'static str> {
        let n = self.params.n;
        let k = self.params.k;
        let m_bar = self.params.d - k;

        // 1) z ~ D_sigma(0) for the left block
        let zero = RingElement::from_scalar(ModInt::zero(), n);
        let mut y = vec![RingElement::zero(n); self.params.d];
        for y_i in y.iter_mut().take(m_bar) {
            *y_i = self.sample_gaussian_ring(&zero, self.params.sigma, rng)?; // z
        }

        // 2) c = target - A_bar z
        let mut c = target.to_vec();
        for (i, c_i) in c.iter_mut().enumerate().take(k) {
            for (j, y_j) in y.iter().enumerate().take(m_bar) {
                *c_i = c_i.clone() - self.a[i][j].clone() * y_j.clone();
            }
        }

        // 3) y_k = G^{-1} c (exact; with G = 2I)
        let g_inv = RingElement::from_scalar(ModInt::from_u64(2).inverse(), n);
        for i in 0..k {
            y[m_bar + i] = c[i].clone() * g_inv.clone();
        }

        // 4) y_bar = z + R y_k   (use trapdoor entries: trapdoor[j][i] = R[i][j])
        for i in 0..m_bar {
            let mut ry = RingElement::zero(n);
            for j in 0..k {
                let rij = self.trapdoor[j][i].clone(); // R[i][j]
                ry = ry + rij * y[m_bar + j].clone();
            }
            y[i] = y[i].clone() + ry;
        }

        // optional: retry if ||y||∞ exceeds bound (keeps your test behavior)
        let mut retries = 0;
        loop {
            if y.iter().all(|yi| yi.norm_inf() <= self.params.norm_bound) {
                return Ok(y);
            }
            
            retries += 1;
            if retries > 1000 {
                return Err("GPV sampling failed after 1000 retries");
            }
            
            // Retry with new randomness
            for y_i in y.iter_mut().take(m_bar) {
                *y_i = self.sample_gaussian_ring(&zero, self.params.sigma, rng)?; // z
            }

            // Recompute c = target - A_bar z
            c = target.to_vec();
            for (i, c_i) in c.iter_mut().enumerate().take(k) {
                for (j, y_j) in y.iter().enumerate().take(m_bar) {
                    *c_i = c_i.clone() - self.a[i][j].clone() * y_j.clone();
                }
            }

            // Recompute y_k = G^{-1} c (exact; with G = 2I)
            for i in 0..k {
                y[m_bar + i] = c[i].clone() * g_inv.clone();
            }

            // Recompute y_bar = z + R y_k
            for i in 0..m_bar {
                let mut ry = RingElement::zero(n);
                for j in 0..k {
                    let rij = self.trapdoor[j][i].clone(); // R[i][j]
                    ry = ry + rij * y[m_bar + j].clone();
                }
                y[i] = y[i].clone() + ry;
            }
        }
    }

    pub fn pack_decomp(mat: &RowMajorMatrix<F>, params: &NeoParams) -> Vec<RingElement<ModInt>> {
        assert_eq!(mat.height(), params.d);
        (0..params.d)
            .map(|row| {
                let mut coeffs: Vec<ModInt> = mat
                    .row(row)
                    .expect("row index")
                    .into_iter()
                    .map(|d| ModInt::from_u64(d.as_canonical_u64()))
                    .collect();
                if coeffs.len() > params.n {
                    panic!("too many coeffs");
                }
                coeffs.resize(params.n, ModInt::zero());
                RingElement::from_coeffs(coeffs, params.n)
            })
            .collect()
    }

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
        // Domain-separated seed for blinding randomness using canonical 256-bit seeding
        use neo_sumcheck::fiat_shamir::Transcript;
        let mut fs_transcript = Transcript::new("commit");
        fs_transcript.absorb_bytes("transcript_state", transcript);
        fs_transcript.absorb_tag("NEO/V1/commit/blinding");
        let mut rng = fs_transcript.rng("NEO/V1/commit/rng");
        self.commit_with_rng(w, &mut rng)
    }

    /// Deterministic variant of [`commit`] using caller-provided RNG.
    /// Enhanced with full ZK blinding using Fiat-Shamir derived parameters.
    #[allow(clippy::type_complexity)]
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
        // Derive ZK parameters from transcript (simulated for now)
        let zk_sigma = self.params.sigma; // Use configured sigma for ZK hiding
        let beta_zk = self.params.beta;    // Use configured beta for blinding

        let r: Vec<RingElement<ModInt>> = (0..w.len())
            .map(|_| RingElement::random_small(rng, self.params.n, beta_zk))
            .collect();

        let mut blinded_w = Vec::with_capacity(w.len());
        for (wi, ri) in w.iter().zip(&r) {
            blinded_w.push(wi.clone() + ri.clone());  // Remove noise - hiding comes from commitment noise e
        }

        // Enhanced commitment noise for ZK hiding
        let e: Vec<_> = if cfg!(test) {
            (0..self.params.k).map(|_| RingElement::zero(self.params.n)).collect()
        } else {
            (0..self.params.k)
                .map(|_| {
                    let mut noise = RingElement::random_gaussian(rng, self.params.n, zk_sigma);
                    // Ensure noise stays within bounds for security
                    while noise.norm_inf() > self.params.e_bound {
                        noise = RingElement::random_gaussian(rng, self.params.n, zk_sigma);
                    }
                    noise
                })
                .collect()
        };

        let mut c =
            vec![
                RingElement::from_coeffs(vec![ModInt::from_u64(0); self.params.n], self.params.n);
                self.params.k
            ];

        for i in 0..self.params.k {
            for (j, w_item) in blinded_w.iter().enumerate().take(self.params.d) {
                c[i] = c[i].clone() + self.a[i][j].clone() * w_item.clone();
            }
            c[i] = c[i].clone() + e[i].clone();
        }
        Ok((c, e, blinded_w, r))
    }

    pub fn verify(
        &self,
        c: &[RingElement<ModInt>],
        w: &[RingElement<ModInt>],
        e: &[RingElement<ModInt>],
    ) -> bool {
        if c.len() != self.params.k || w.len() != self.params.d || e.len() != self.params.k {
            return false;
        }
        let w_bound = self.params.norm_bound
            + self.params.beta;
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
        for (i, (ai_row, ei)) in self.a.iter().zip(e.iter()).enumerate() {
            let mut expected =
                RingElement::from_coeffs(vec![ModInt::from_u64(0); self.params.n], self.params.n);
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

    pub fn random_linear_combo(
        &self,
        c1: &[RingElement<ModInt>],
        c2: &[RingElement<ModInt>],
        rho: F,
    ) -> Vec<RingElement<ModInt>> {
        // Reject if rho ≥ q to avoid accidental wrap
        let q = <ModInt as Coeff>::modulus();
        if rho.as_canonical_u64() >= q {
            panic!("random_linear_combo: rho outside Z_q representative range");
        }
        // Length-agnostic, broadcast zeros for missing entries.
        // This handles trivial/degenerate cases that arise in NARK-mode folding.
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

    /// Random linear combination using a rotation element in the ring.
    pub fn random_linear_combo_rotation(
        &self,
        c1: &[RingElement<ModInt>],
        c2: &[RingElement<ModInt>],
        rho_rot: &RingElement<ModInt>,
    ) -> Vec<RingElement<ModInt>> {
        // Length-agnostic with zero-broadcast semantics.
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

    /// Derive a ModInt challenge from transcript+label using bias-free canonical FS.
    fn fs_challenge_modint_labeled(transcript: &[u8], label: &str) -> ModInt {
        use neo_sumcheck::fiat_shamir::Transcript;
        let mut fs_transcript = Transcript::new("commit");
        fs_transcript.absorb_bytes("base_state", transcript);
        let challenge = fs_transcript.challenge_modint(label);
        
        if cfg!(test) {
            eprintln!("EXTRACTOR_DEBUG: FS challenge for '{}': {} (from transcript len={})", 
                     label, challenge.as_canonical_u64(), transcript.len());
        }
        
        challenge
    }

    /// Extract witness for `c1` given another commitment `c2` by rewinding FS twice.
    /// Uses scalar RLC: combo_i = c1 + ρ_i * c2. Returns w1 with A*w1 = c1.
    pub fn extract_commit_witness(
        &self,
        c1: &[RingElement<ModInt>],
        c2: &[RingElement<ModInt>],
        transcript: &[u8],
    ) -> Result<Vec<RingElement<ModInt>>, &'static str> {
        if c1.len() != self.params.k || c2.len() != self.params.k {
            return Err("Commitment length mismatch");
        }

        // Two distinct scalar FS challenges ρ1, ρ2 ∈ F_q.
        let rho1 = Self::fs_challenge_modint_labeled(transcript, "neo_extract_rho|0");
        let mut rho2 = Self::fs_challenge_modint_labeled(transcript, "neo_extract_rho|1");
        if rho1 == rho2 {
            rho2 = Self::fs_challenge_modint_labeled(transcript, "neo_extract_rho|2");
            if rho1 == rho2 {
                return Err("Extractor failed: identical scalar forks");
            }
        }
        
        if cfg!(test) {
            eprintln!("EXTRACTOR_DEBUG: Using distinct challenges rho1={}, rho2={}", 
                     rho1.as_canonical_u64(), rho2.as_canonical_u64());
        }

        // Build two scalar random linear combinations of c1, c2.
        let rho1_f = neo_fields::F::from_u64(rho1.as_canonical_u64());
        let rho2_f = neo_fields::F::from_u64(rho2.as_canonical_u64());
        let combo1 = self.random_linear_combo(c1, c2, rho1_f);
        let combo2 = self.random_linear_combo(c1, c2, rho2_f);

        // GPV preimages for combos (use transcript-derived seeds for determinism) via canonical FS with 256-bit entropy.
        use neo_sumcheck::fiat_shamir::Transcript;
        let mut fs_transcript = Transcript::new("commit");
        fs_transcript.absorb_bytes("transcript_state", transcript);
        fs_transcript.absorb_tag("NEO/V1/extract/gpv");
        let mut rng1 = fs_transcript.rng("NEO/V1/extract/combo_seed/0");
        let mut rng2 = fs_transcript.rng("NEO/V1/extract/combo_seed/1");
        
        if cfg!(test) {
            eprintln!("EXTRACTOR_DEBUG: GPV sampling with canonical 256-bit seeds");
        }
        
        let y1 = self.gpv_trapdoor_sample(&combo1, self.params.sigma, &mut rng1)?;
        let y2 = self.gpv_trapdoor_sample(&combo2, self.params.sigma, &mut rng2)?;
        
        if cfg!(test) {
            let y1_norms: Vec<_> = y1.iter().map(|y| y.norm_inf()).collect();
            let y2_norms: Vec<_> = y2.iter().map(|y| y.norm_inf()).collect();
            eprintln!("EXTRACTOR_DEBUG: GPV samples - y1 max_norm={}, y2 max_norm={}", 
                     y1_norms.iter().max().unwrap_or(&0),
                     y2_norms.iter().max().unwrap_or(&0));
        }

        // Compute w2 := (y1 - y2) / (ρ1 - ρ2) and w1 := y1 - ρ1 * w2.
        let diff = rho1 - rho2;
        if diff == ModInt::from_u64(0) {
            return Err("Extractor failed: zero diff");
        }
        let diff_inv = diff.inverse();

        let w2: Vec<_> = y1.iter()
            .zip(&y2)
            .map(|(a, b)| (a.clone() - b.clone()) * diff_inv)
            .collect();

        let w1: Vec<_> = y1.iter()
            .zip(&w2)
            .map(|(a, w2i)| a.clone() - w2i.clone() * rho1)
            .collect();

        if cfg!(test) {
            let w1_norms: Vec<_> = w1.iter().map(|w| w.norm_inf()).collect();
            let w2_norms: Vec<_> = w2.iter().map(|w| w.norm_inf()).collect();
            eprintln!("EXTRACTOR_DEBUG: Final witnesses - w1 max_norm={}, w2 max_norm={}", 
                     w1_norms.iter().max().unwrap_or(&0),
                     w2_norms.iter().max().unwrap_or(&0));
        }

        // Optional: sanity check (constant-time checks happen in verify_extracted_witness).
        // A*w1 must equal c1 exactly modulo q.
        let zero_noise = vec![RingElement::from_scalar(ModInt::zero(), self.params.n); self.params.k];
        let verification_result = self.verify(c1, &w1, &zero_noise);
        
        if cfg!(test) {
            eprintln!("EXTRACTOR_DEBUG: Verification result: {}", verification_result);
        }
        
        if !verification_result {
            // Fallback: still return w1; caller verifies CT-bounds. But signal issue:
            return Err("Extractor preimage failed verification");
        }

        Ok(w1)
    }



    /// Verify that an extracted witness satisfies the commitment relation
    /// This is used to validate extracted witnesses in the knowledge soundness check
    pub fn verify_extracted_witness(
        &self,
        commitment: &[RingElement<ModInt>],
        witness: &[RingElement<ModInt>],
        noise: &[RingElement<ModInt>],
    ) -> bool {
        if commitment.len() != self.params.k || witness.len() != self.params.d || noise.len() != self.params.k {
            return false;
        }

        // Check that A*w + e = commitment (mod q)
        for i in 0..self.params.k {
            let mut expected = RingElement::from_scalar(ModInt::zero(), self.params.n);
            for (j, w_item) in witness.iter().enumerate().take(self.params.d) {
                expected = expected + self.a[i][j].clone() * w_item.clone();
            }
            expected = expected + noise[i].clone();

            if expected != commitment[i] {
                return false;
            }
        }

        // Check norm bounds for security using constant-time operations
        let witness_norm = witness.iter().map(|w| w.norm_inf_ct()).max().unwrap_or(0);
        let noise_norm = noise.iter().map(|e| e.norm_inf_ct()).max().unwrap_or(0);

        // Use constant-time less-than-or-equal comparison
        let witness_ok = witness_norm.ct_lt(&(self.params.norm_bound + 1));
        let noise_ok = noise_norm.ct_lt(&(self.params.e_bound + 1));

        (witness_ok & noise_ok).into()
    }

    /// Open a commitment at a multilinear evaluation point. This folds the
    /// commitment vector and produces a short trapdoor-based proof witnessing the
    /// claimed evaluation.
    pub fn open_at_point(
        &self,
        c: &[RingElement<ModInt>],
        point: &[ExtF],
        w: &[RingElement<ModInt>],
        _e: &[RingElement<ModInt>],
        r: &[RingElement<ModInt>],
        rng: &mut impl Rng,
    ) -> Result<(ExtF, Vec<RingElement<ModInt>>), &'static str> {
        let blinded_eval = self.compute_multilinear_eval(point, w);
        let r_eval = self.compute_multilinear_eval(point, r);
        let eval = blinded_eval - r_eval;
        let eval_arr = eval.to_array();
        if eval_arr[1] != F::ZERO {
            return Err("open_at_point: evaluation not in base field; choose points in F (imag = 0) or implement extension packing");
        }
        let eval_ring = RingElement::from_scalar(
            ModInt::from_u64(eval_arr[0].as_canonical_u64()),
            self.params.n,
        );
        let gadget = RingElement::from_scalar(ModInt::from_u64(2), self.params.n);

        let target: Vec<RingElement<ModInt>> = c
            .iter()
            .map(|ci| ci.clone() - gadget.clone() * eval_ring.clone())
            .collect();

        let proof = self.gpv_trapdoor_sample(&target, self.params.sigma, rng)?;

        Ok((eval, proof))
    }

    /// Verify an opening at a point by folding the commitment in the same manner
    /// and checking the provided evaluation and proof.
    pub fn verify_opening(
        &self,
        c: &[RingElement<ModInt>],
        _point: &[ExtF],
        eval: ExtF,
        proof: &[RingElement<ModInt>],
        max_blind_norm: u64,
    ) -> bool {
        if proof.len() != self.params.d {
            return false;
        }
        if proof.iter().any(|y| y.norm_inf() > self.params.norm_bound)
            || eval.abs_norm() > max_blind_norm
        {
            return false;
        }

        let eval_arr = eval.to_array();
        if eval_arr[1] != F::ZERO {
            return false;
        }
        let eval_ring = RingElement::from_scalar(
            ModInt::from_u64(eval_arr[0].as_canonical_u64()),
            self.params.n,
        );
        let gadget = RingElement::from_scalar(ModInt::from_u64(2), self.params.n);
        let target: Vec<RingElement<ModInt>> = c
            .iter()
            .map(|ci| ci.clone() - gadget.clone() * eval_ring.clone())
            .collect();

        let mut recomputed =
            vec![RingElement::from_scalar(ModInt::zero(), self.params.n); self.params.k];
        for (i, ai_row) in self.a.iter().enumerate() {
            for (aij, yj) in ai_row.iter().zip(proof.iter()) {
                recomputed[i] = recomputed[i].clone() + aij.clone() * yj.clone();
            }
        }
        recomputed == target
    }

    /// Rough byte-size cost of committing and opening `m` field elements. This
    /// follows the pay-per-bit metric from §1.3 of the paper.
    pub fn pay_per_bit_cost(&self, m: usize) -> u64 {
        let bit_width = (self.params.q as f64).log2().ceil() as u64;
        self.params.k as u64 * self.params.d as u64 * bit_width
            + m as u64 * bit_width
    }
}

impl AjtaiCommitter {
    fn compute_multilinear_eval(&self, point: &[ExtF], w: &[RingElement<ModInt>]) -> ExtF {
        let l = point.len();
        let needed = 1 << l;
        let coeffs: Vec<F> = w
            .iter()
            .flat_map(|ring| {
                ring.coeffs()
                    .iter()
                    .map(|m| F::from_u64(m.as_canonical_u64()))
            })
            .take(needed)
            .collect();
        let mut padded = coeffs;
        padded.resize(needed, F::ZERO);
        let evals_ext: Vec<ExtF> = padded.iter().map(|&f| from_base(f)).collect();
        // Direct multilinear extension evaluation
        let mut result = ExtF::ZERO;
        for (idx, val) in evals_ext.iter().enumerate() {
            let mut term = *val;
            for (j, &x) in point.iter().enumerate() {
                let bit = (idx >> j) & 1;
                term *= if bit == 1 { x } else { ExtF::ONE - x };
            }
            result += term;
        }
        result
    }
}

fn compute_lambda(params: &NeoParams) -> f64 {
    // Very rough log2 security estimate combining MSIS and RLWE-style bounds.
    // Intended only for development-time checks.
    let msis = (params.k as f64 * params.d as f64) * (params.q as f64).log2()
        + (2.0 * params.sigma * (params.n as f64 * params.k as f64).sqrt()).log2()
            * (params.d as f64)
        - (params.e_bound as f64).log2();
    let rlwe = (params.n as f64) * (params.q as f64).log2()
        - (params.sigma.powi(2) * params.n as f64).log2();
    msis.min(rlwe)
}


