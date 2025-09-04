//! Strong sampling set C ⊆ S (rotation-matrix ring) and Fiat–Shamir sampling.
//! Matches Neo §3.4 definition and usage in Π_RLC. See also Theorem 3 (expansion factor).
//! (Backend defines the element type `S` and how it acts on commitments/matrices/vectors.)
//!
//! Security note: caller must choose C that satisfies Neo's invertibility/expansion requirements.
//!
//! Domain separation tags used here are public and stable.
//!
//! Reference: Neo paper §§3.4, 4.5. (file-cited externally)

use crate::transcript::FoldTranscript;
use p3_field::PrimeField64;

/// Public, stable labels for Fiat–Shamir.
pub mod ds {
    pub const RLC: &[u8] = b"neo.rlc.v1/verify_linear";
    pub const DEC: &[u8] = b"neo.dec.v1/verify_decomposition";
    pub const CCS: &[u8] = b"neo.ccs.v1/eval_check";
}

/// A strong sampling set for challenges (rotation-operators in S).
#[derive(Clone, Debug)]
pub struct StrongSamplingSet<S> {
    elems: Vec<S>,
}

impl<S: Clone> StrongSamplingSet<S> {
    pub fn new(elems: Vec<S>) -> Self {
        assert!(!elems.is_empty(), "C must be non-empty");
        Self { elems }
    }

    #[inline]
    pub fn len(&self) -> usize { 
        self.elems.len() 
    }
    
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    /// Sample exactly k challenges ρ_i ∈ C by FS with bias-free rejection sampling.
    /// Uses rejection sampling to eliminate modulo bias when reducing to index range.
    pub fn sample_k(&self, t: &mut FoldTranscript, label: &[u8], k: usize) -> Vec<S> {
        let n = self.elems.len() as u64;
        assert!(n > 0, "Strong set cannot be empty");
        
        // SECURITY: Prevent infinite loop if set size exceeds field modulus
        // CRITICAL FIX: Use field modulus, not u64::MAX, for proper bias correction
        let q = neo_math::F::ORDER_U64; // challenge_f() returns values in [0, q)
        assert!(n <= q, 
            "Strong set size {} exceeds field modulus {}, would cause infinite rejection loop", 
            n, q);
        
        let mut out = Vec::with_capacity(k);
        let bound = q - (q % n);        // largest multiple of n <= q
        
        for i in 0..k {
            // Use our transcript's challenge mechanism with domain separation
            t.absorb_bytes(&[label, b"/rho/", &(i as u64).to_le_bytes()].concat());
            // Rejection sampling to avoid modulo bias
            loop {
                let f = t.challenge_f().as_canonical_u64(); // uniform in [0, q)
                if f < bound {
                    let idx = (f % n) as usize;
                    out.push(self.elems[idx].clone());
                    break;
                }
                // If rejected, absorb a salt and try again to get fresh randomness
                t.absorb_bytes(b"/rejection/");
            }
        }
        out
    }
}

/// Canonical verification errors across reductions.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum VerificationError {
    #[error("linear relation on commitments failed at term {0}")]
    LinearCommit(usize),
    #[error("linear relation on X failed at row {0}, col {1}")]
    LinearX(usize, usize),
    #[error("linear relation on y[{0}] failed at coordinate {1}")]
    LinearY(usize, usize),
    #[error("decomposition identity failed for commitments (b^i * c_i sum != c)")]
    DecompCommit,
    #[error("decomposition identity failed for X (b^i * X_i rows sum != X)")]
    DecompX,
    #[error("decomposition identity failed for y[{0}] (b^i * y_i sum != y)")]
    DecompY(usize),
    #[error("CCS evaluation check failed (v != Q(alpha', r'))")]
    CcsEvalMismatch,
}
