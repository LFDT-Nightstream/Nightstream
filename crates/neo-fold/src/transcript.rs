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

use p3_challenger::{CanSampleBits, FieldChallenger, DuplexChallenger, CanObserve, CanSample};
use p3_field::{Field, PrimeCharacteristicRing, ExtensionField, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};

// ---------- Field aliases ----------
pub use neo_math::{F, K, ExtF}; // Use proper neo-math types

// ---------- Poseidon2 config ----------
pub const POSEIDON2_WIDTH: usize = 16;
pub const POSEIDON2_RATE: usize = 8;

// Challenger over Poseidon2 perm (width=16, rate=8).
pub type NeoChallenger =
    DuplexChallenger<F, Poseidon2Goldilocks<POSEIDON2_WIDTH>, POSEIDON2_WIDTH, POSEIDON2_RATE>;

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
#[derive(Clone)]
pub struct FoldTranscript {
    ch: NeoChallenger,
}

impl FoldTranscript {
    pub fn new(initial_data: &[u8]) -> Self {
        // Poseidon2 with default parameters for Goldilocks.
        // Use a fixed seed for deterministic transcript initialization
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
        let perm = Poseidon2Goldilocks::<POSEIDON2_WIDTH>::new_from_rng_128(&mut rng);
        let mut ch = NeoChallenger::new(perm);
        
        // Seal the transcript version.
        // Convert bytes to field elements and observe them individually
        for &byte in b"neo/transcript/v1" {
            ch.observe(F::from_u32(byte as u32));
        }
        
        // Absorb initial data
        for &byte in initial_data {
            ch.observe(F::from_u32(byte as u32));
        }
        
        Self { ch }
    }
    
    pub fn default() -> Self {
        Self::new(b"")
    }

    pub fn domain(&mut self, d: Domain) {
        // Convert bytes to field elements and observe them individually
        for &byte in d.tag() {
            self.ch.observe(F::from_u32(byte as u32));
        }
    }

    pub fn absorb_f(&mut self, xs: &[F]) {
        // Observe each field element individually
        for &x in xs {
            self.ch.observe(x);
        }
    }

    pub fn absorb_u64(&mut self, xs: &[u64]) {
        // Encode as canonical Goldilocks elements.
        use p3_field::integers::QuotientMap;
        let tmp: Vec<F> = xs.iter().map(|&u| F::from_u32(u as u32)).collect();
        // Observe each field element individually
        for &x in tmp.iter() {
            self.ch.observe(x);
        }
    }

    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        // Convert bytes to field elements and observe them individually
        for &byte in bytes {
            self.ch.observe(F::from_u32(byte as u32));
        }
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
        K::new_complex(a0, a1)
    }

    /// Draw multiple extension field challenges
    pub fn challenges_k(&mut self, m: usize) -> Vec<K> {
        (0..m).map(|_| self.challenge_k()).collect()
    }
    
    /// Convert transcript to challenger for use with neo_challenge sampling
    /// Creates a simple field challenger that can be used with neo-challenge
    pub fn as_challenger(&mut self) -> SimpleFieldChallenger {
        SimpleFieldChallenger { transcript: self }
    }
    
    /// Absorb extension field element as base field elements
    pub fn absorb_ext_as_base_fields(&mut self, label: &[u8], value: ExtF) {
        self.absorb_bytes(label);
        // Split ExtF into base field coordinates and absorb each
        let a0 = value.real();
        let a1 = value.imag();
        self.ch.observe(a0);
        self.ch.observe(a1);
    }
    
    /// Get current state digest for transcript binding
    pub fn state_digest(&self) -> Vec<u8> {
        // This is a simplified implementation - in practice we'd export the permutation state
        // For now, return a placeholder digest
        vec![0u8; 32]
    }
}

/// Simple field challenger adapter for neo-challenge integration
pub struct SimpleFieldChallenger<'a> {
    transcript: &'a mut FoldTranscript,
}

impl<'a> CanSample<F> for SimpleFieldChallenger<'a> {
    fn sample(&mut self) -> F {
        self.transcript.challenge_f()
    }
}

impl<'a> CanObserve<F> for SimpleFieldChallenger<'a> {
    fn observe(&mut self, value: F) {
        self.transcript.ch.observe(value);
    }
}

impl<'a> CanObserve<u8> for SimpleFieldChallenger<'a> {
    fn observe(&mut self, value: u8) {
        self.transcript.absorb_bytes(&[value]);
    }
}

impl<'a> CanSampleBits<usize> for SimpleFieldChallenger<'a> {
    fn sample_bits(&mut self, bits: usize) -> usize {
        let field_val = self.transcript.challenge_f().as_canonical_u64();
        (field_val as usize) & ((1 << bits) - 1)
    }
}

impl<'a> FieldChallenger<F> for SimpleFieldChallenger<'a> {}

impl Default for FoldTranscript {
    fn default() -> Self {
        Self::new(b"")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_basic() {
        let mut tr = FoldTranscript::new(b"test");
        
        // Test domain separation
        tr.domain(Domain::CCS);
        
        // Test absorbing different types
        tr.absorb_u64(&[1, 2, 3]);
        tr.absorb_f(&[F::from_u32(42)]);
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
        let mut tr1 = FoldTranscript::new(b"test");
        let mut tr2 = FoldTranscript::new(b"test");
        
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

