//! Poseidon2-backed Fiat-Shamir transcript for neo-fold
//!
//! This provides domain-separated challenges using Poseidon2 over Goldilocks,
//! ensuring consistency across the entire pipeline (folding + FRI).
//!
//! Key features:
//! - Single transcript used across all protocols (Π_CCS, Π_RLC, Π_DEC, sum-check)
//! - Domain separation for security
//! - Support for both base field F and extension field K = F_{q^2} challenges
//! - Compatible with p3-FRI's challenger interface

#![allow(unused_imports)]

use p3_challenger::{CanSampleBits, FieldChallenger};
use p3_field::{AbstractField, ExtensionField, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_poseidon2::Poseidon2;
use p3_symmetric::DuplexChallenger;

// ---------- Field aliases ----------
pub type F = Goldilocks; // q = 2^64 - 2^32 + 1

// TODO: Import from neo-math when available
// For now, we'll define a simple extension field K = F_{q^2}
// This should match neo-math's K type exactly
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K {
    pub c0: F, // constant term
    pub c1: F, // coefficient of the extension element
}

impl K {
    pub fn from_base_slice(slice: &[F]) -> Option<Self> {
        if slice.len() >= 2 {
            Some(K { c0: slice[0], c1: slice[1] })
        } else {
            None
        }
    }
}

// ---------- Poseidon2 config ----------
pub const POSEIDON2_WIDTH: usize = 16;
pub const POSEIDON2_RATE: usize = 8;

// Challenger over Poseidon2 perm (width=16, rate=8).
pub type Perm = Poseidon2<F, POSEIDON2_WIDTH>;
pub type NeoChallenger = DuplexChallenger<F, Perm, POSEIDON2_RATE>;

// Domain separation tags (keep them stable; recorded in the transcript header).
#[derive(Copy, Clone)]
pub enum Domain {
    CCS,
    Rlc,
    Dec,
    Sumcheck,
    FriCommit,
    FriQuery,
}

impl Domain {
    fn tag(self) -> &'static [u8] {
        match self {
            Domain::CCS       => b"neo/v1/pi_ccs",
            Domain::Rlc       => b"neo/v1/pi_rlc", 
            Domain::Dec       => b"neo/v1/pi_dec",
            Domain::Sumcheck  => b"neo/v1/sumcheck",
            Domain::FriCommit => b"neo/v1/fri/commit",
            Domain::FriQuery  => b"neo/v1/fri/query",
        }
    }
}

/// Transcript facade providing domain-separated challenges
pub struct FoldTranscript {
    ch: NeoChallenger,
}

impl FoldTranscript {
    pub fn new() -> Self {
        // Poseidon2 with default parameters for Goldilocks.
        let perm = Perm::new();
        let mut ch = NeoChallenger::new(perm);
        // Seal the transcript version.
        ch.observe_bytes(b"neo/transcript/v1");
        Self { ch }
    }

    pub fn domain(&mut self, d: Domain) {
        self.ch.observe_bytes(d.tag());
    }

    pub fn absorb_f(&mut self, xs: &[F]) {
        self.ch.observe_slice(xs);
    }

    pub fn absorb_u64(&mut self, xs: &[u64]) {
        // Encode as canonical Goldilocks elements.
        use p3_field::integers::QuotientMap;
        let tmp: Vec<F> = xs.iter().map(|&u| F::from_canonical_checked(u).unwrap_or(F::ZERO)).collect();
        self.ch.observe_slice(&tmp);
    }

    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.ch.observe_bytes(bytes);
    }

    /// Draw a base-field challenge in F.
    pub fn challenge_f(&mut self) -> F {
        self.ch.sample() // FieldChallenger provides field sampling
    }

    /// Draw `m` base-field challenges.
    pub fn challenges_f(&mut self, m: usize) -> Vec<F> {
        (0..m).map(|_| self.challenge_f()).collect()
    }

    /// Draw a single extension-field challenge in K = F_{q^2}.
    /// Constructs K from two base-field challenges (lexicographic).
    pub fn challenge_k(&mut self) -> K {
        let a0 = self.challenge_f();
        let a1 = self.challenge_f();
        K { c0: a0, c1: a1 }
    }

    /// Draw multiple extension field challenges
    pub fn challenges_k(&mut self, m: usize) -> Vec<K> {
        (0..m).map(|_| self.challenge_k()).collect()
    }
}

impl Default for FoldTranscript {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_basic() {
        let mut tr = FoldTranscript::new();
        
        // Test domain separation
        tr.domain(Domain::CCS);
        
        // Test absorbing different types
        tr.absorb_u64(&[1, 2, 3]);
        tr.absorb_f(&[F::from_canonical_checked(42).unwrap()]);
        tr.absorb_bytes(b"test_data");
        
        // Test challenge generation
        let f_challenges = tr.challenges_f(3);
        assert_eq!(f_challenges.len(), 3);
        
        let k_challenge = tr.challenge_k();
        println!("Extension challenge: {:?}", k_challenge);
        
        println!("Transcript test completed successfully");
    }

    #[test]
    fn test_domain_separation() {
        let mut tr1 = FoldTranscript::new();
        let mut tr2 = FoldTranscript::new();
        
        // Same input, different domains
        tr1.domain(Domain::CCS);
        tr2.domain(Domain::Rlc);
        
        tr1.absorb_u64(&[1, 2, 3]);
        tr2.absorb_u64(&[1, 2, 3]);
        
        let c1 = tr1.challenge_f();
        let c2 = tr2.challenge_f();
        
        // Should be different due to domain separation
        assert_ne!(c1, c2);
        println!("Domain separation working: {} != {}", c1.as_canonical_u64(), c2.as_canonical_u64());
    }
}

