//! Strong-sampler for Neo: sample ρ = rot(a) with short coeffs, enforce invertible differences,
//! compute expansion bound T, and expose domain-separated APIs.
//!
//! MUST (paper §3.4, §4.3): C = {rot(a)} with a from small-coeff set C_R; pairwise differences invertible;
//! transcript-seeded; record expansion T. SHOULD: metrics for T and failure-rate tests.
//!
//! References: Neo ePrint 2025/294 (Defs. 10, 12, 14; Thms. 3, 7), and LS18 for short-invertibility heuristics.

mod metrics;

use core::fmt::Debug;
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks as Fq;
use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// Bring the rotation map from R_q to S ⊂ F_q^{d×d}.
use neo_math::{SAction, cf_inv}; // Use SAction for rotation matrices
use neo_math::{D}; // dimension constant

// ---------- Public types ----------

/// Configuration of the strong set C ⊂ S and its coefficient source C_R ⊂ R_q.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrongSetConfig {
    /// Cyclotomic index η (e.g., 81). Used only to compute φ(η) for T.
    pub eta: u32,
    /// Dimension d = φ(η).
    pub d: usize,
    /// Coefficients are sampled from [-coeff_bound, coeff_bound] ⊂ ℤ reduced into F_q.
    pub coeff_bound: i32,
    /// Domain separator used before sampling.
    pub domain_sep: &'static [u8],
    /// Maximum resampling attempts for ensuring invertible pairwise differences.
    pub max_resamples: usize,
}

impl StrongSetConfig {
    /// Euler's totient φ(η).
    pub fn phi_eta(&self) -> u64 { phi(self.eta as u64) }
    /// Expansion bound T ≤ 2·φ(η)·H where H = coeff_bound.
    pub fn expansion_upper_bound(&self) -> u64 {
        2u64 * self.phi_eta() * (self.coeff_bound.unsigned_abs() as u64)
    }
}

/// A sampled challenge element ρ = rot(a) with its underlying coefficient vector `a` in coefficient form.
#[derive(Clone, Debug)]
pub struct Rho {
    /// a in coefficient form (length d) over F_q.
    pub coeffs: Vec<Fq>,
    /// ρ = rot(a) ∈ F_q^{d×d}.
    pub matrix: RowMajorMatrix<Fq>,
}

#[derive(Debug, Error)]
pub enum ChallengeError {
    #[error("failed to find invertible pairwise differences after {attempts} attempts")]
    NonInvertible { attempts: usize },
    #[error("dimension mismatch: expected d={expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

// ---------- API (MUST): domain-separated, transcript-seeded sampling ----------

/// Domain-separated sampler for a single ρ = rot(a).
///
/// `C` can be any `FieldChallenger<Fq>` (Poseidon2-based `DuplexChallenger` or hash-based `HashChallenger`).
pub fn sample_rho<C>(chal: &mut C, cfg: &StrongSetConfig) -> Rho
where
    C: FieldChallenger<Fq> + CanObserve<u8> + CanSampleBits<usize>,
{
    // Domain separation.
    chal.observe_slice(cfg.domain_sep);

    // Sample coefficients a_i ∈ [-H..H] mapped into F_q.
    let mut a = Vec::with_capacity(cfg.d);
    let width = (2 * cfg.coeff_bound + 1) as usize;
    let needed_bits = (usize::BITS - (width - 1).leading_zeros()) as usize;

    for _ in 0..cfg.d {
        // Use rejection sampling to avoid modulo bias
        let x = loop {
            let candidate = chal.sample_bits(needed_bits);
            if candidate < width {
                break candidate;
            }
            // Reject and try again - this eliminates modulo bias
        };
        let z = (x as i32) - cfg.coeff_bound;
        a.push(int_to_fq(z));
    }

    let mat = rot_from_coeffs(&a);
    Rho { coeffs: a, matrix: mat }
}

/// Sample k+1 elements ρ₁,...,ρ_{k+1} with **pairwise invertible differences**, resampling if needed (MUST).
///
/// This is the object used by Π_RLC (paper §4.5). Returns also the computed **T** bound from `cfg`.
pub fn sample_kplus1_invertible<C>(
    chal: &mut C,
    cfg: &StrongSetConfig,
    k_plus_one: usize,
) -> Result<(Vec<Rho>, u64), ChallengeError>
where
    C: FieldChallenger<Fq> + CanObserve<u8> + CanSampleBits<usize>,
{
    let mut attempts = 0usize;

    loop {
        attempts += 1;
        // fresh batch of ρ's
        let rhos: Vec<Rho> = (0..k_plus_one).map(|_| sample_rho(chal, cfg)).collect();

        // Check pairwise invertibility: rot(a_i - a_j) invertible for all i≠j (Theorem 7 ⇒ rot linear; matrix invertibility ↔ ring invertibility).
        if all_pairwise_differences_invertible(&rhos, cfg.d) {
            return Ok((rhos, cfg.expansion_upper_bound()));
        }

        if attempts >= cfg.max_resamples {
            return Err(ChallengeError::NonInvertible { attempts });
        }

        // Re-seed domain to avoid cycles; light domain bump.
        chal.observe_slice(b"neo.challenge.reroll");
    }
}

// ---------- SHOULD: metrics helpers (empirical expansion, failure rate) ----------

pub use metrics::{empirical_expansion_stats, ExpansionStats};

// ---------- Internals ----------

#[inline]
fn int_to_fq(z: i32) -> Fq {
    // Map signed integer into F_q canonically.
    if z >= 0 {
        Fq::from_u64(z as u64)
    } else {
        let neg = (-z) as u64;
        Fq::ZERO - Fq::from_u64(neg)
    }
}

fn rot_from_coeffs(coeffs: &[Fq]) -> RowMajorMatrix<Fq> {
    // Convert coefficients to ring element and then to rotation matrix via SAction
    // Build Rq from coefficients
    let mut rq_coeffs = [Fq::ZERO; D];
    for (i, &c) in coeffs.iter().enumerate().take(D) {
        rq_coeffs[i] = c;
    }
    let rq = cf_inv(rq_coeffs);
    
    // Get rotation matrix via SAction
    let s_action = SAction::from_ring(rq);
    let dense_matrix = s_action.to_matrix();
    
    // Convert DenseMatrix to RowMajorMatrix 
    RowMajorMatrix::new(dense_matrix.values, dense_matrix.width)
}

fn all_pairwise_differences_invertible(rhos: &[Rho], d: usize) -> bool {
    for i in 0..rhos.len() {
        for j in (i + 1)..rhos.len() {
            // diff coeffs
            let mut diff = vec![Fq::ZERO; d];
            diff.iter_mut().enumerate().for_each(|(t, elem)| {
                *elem = rhos[i].coeffs[t] - rhos[j].coeffs[t];
            });
            // rot(a_i - a_j)
            let m = rot_from_coeffs(&diff);
            if !is_invertible_row_major(m, d) {
                return false;
            }
        }
    }
    true
}

/// Gauss-Jordan over F_q to test invertibility. O(d^3), fine for d≈54 (η=81, φ(η)=54).
fn is_invertible_row_major(a: RowMajorMatrix<Fq>, d: usize) -> bool {
    let width = a.width;
    debug_assert_eq!(width, d);
    let mut data = a.values;

    // elimination
    for col in 0..d {
        // find pivot
        let mut pivot = None;
        for r in col..d {
            if !data[r * d + col].is_zero() { pivot = Some(r); break; }
        }
        let piv = match pivot { Some(r) => r, None => return false };
        
        // swap to row 'col'
        if piv != col {
            for c in 0..d {
                data.swap(col * d + c, piv * d + c);
            }
        }
        
        // scale pivot row to 1
        let pivot_val = data[col * d + col];
        let inv = pivot_val.inverse();
        for c in col..d { 
            data[col * d + c] *= inv;
        }
        
        // eliminate other rows
        for r in 0..d {
            if r == col { continue; }
            let f = data[r * d + col];
            if f.is_zero() { continue; }
            for c in col..d {
                let pivot_coeff = data[col * d + c];
                data[r * d + c] -= f * pivot_coeff;
            }
        }
    }
    true
}

/// Euler totient (small η).
fn phi(mut n: u64) -> u64 {
    let mut result = n;
    let mut p = 2u64;
    while p * p <= n {
        if n % p == 0 {
            while n % p == 0 { n /= p; }
            result = result / p * (p - 1);
        }
        p += 1;
    }
    if n > 1 { result = result / n * (n - 1); }
    result
}

// ---------- Defaults for GL-128 profile ----------

/// Default config matching STRUCTURE.md: η=81 (φ(η)=54), H=2 ⇒ T=216.
pub const DEFAULT_STRONGSET: StrongSetConfig = StrongSetConfig {
    eta: 81,
    d: 54,
    coeff_bound: 2,
    domain_sep: b"neo.challenge.rlc.rho.v1",
    max_resamples: 16,
};