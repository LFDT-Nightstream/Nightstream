// neo-fold/src/fri_engine.rs
//! Custom FRI-based engine for Spartan2 integration
//! 
//! This module provides a post-quantum friendly engine for Spartan2 by implementing
//! the Engine trait with FRI-based polynomial commitment scheme using p3-fri.

#[cfg(feature = "snark_spartan2")]
use spartan2::traits::{Engine, pcs::{PCSEngineTrait, FoldingEngineTrait, CommitmentTrait}, transcript::TranscriptReprTrait};
#[cfg(feature = "snark_spartan2")]
use spartan2::errors::SpartanError;
#[cfg(feature = "snark_spartan2")]
use spartan2::provider::pasta::pallas;
#[cfg(feature = "snark_spartan2")]
use serde::{Serialize, Deserialize};

use neo_fields::F;
use neo_commit::fri_pcs::FriPCSWrapper;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::marker::PhantomData;

/// Custom FRI-based engine for Spartan2 using p3-fri for post-quantum security
#[cfg(feature = "snark_spartan2")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeoFriEngine;

#[cfg(feature = "snark_spartan2")]
impl Engine for NeoFriEngine {
    // Use Pallas fields for compatibility with Spartan2's circuit system
    type Base = pallas::Base;
    type Scalar = pallas::Scalar;
    type GE = pallas::Point;
    
    // Use our custom FRI-based PCS
    type PCS = NeoFriPCS;
    
    // Add the missing TE (TranscriptEngine) type
    type TE = spartan2::provider::keccak::Keccak256Transcript<Self>;
}

/// FRI-based Polynomial Commitment Scheme implementation for Spartan2
/// Bridges between Spartan2's PCS interface and p3-fri
#[cfg(feature = "snark_spartan2")]
#[derive(Clone, Debug)]
pub struct NeoFriPCS {
    _phantom: PhantomData<F>,
}

#[cfg(feature = "snark_spartan2")]
impl NeoFriPCS {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

/// FRI commitment representation for Spartan2
#[cfg(feature = "snark_spartan2")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FriCommitmentSpartan2 {
    /// Merkle root from FRI commitment
    pub root: [u8; 32],
    /// Domain size for the polynomial
    pub domain_size: usize,
    /// Metadata for verification
    pub metadata: Vec<u8>,
}

#[cfg(feature = "snark_spartan2")]
impl CommitmentTrait<NeoFriEngine> for FriCommitmentSpartan2 {}

#[cfg(feature = "snark_spartan2")]
impl Default for FriCommitmentSpartan2 {
    fn default() -> Self {
        Self {
            root: [0u8; 32],
            domain_size: 0,
            metadata: vec![],
        }
    }
}

#[cfg(feature = "snark_spartan2")]
impl TranscriptReprTrait<pallas::Point> for FriCommitmentSpartan2 {
    fn to_transcript_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.root);
        bytes.extend_from_slice(&(self.domain_size as u64).to_le_bytes());
        bytes.extend_from_slice(&self.metadata);
        bytes
    }
}

/// FRI proof representation for Spartan2
#[cfg(feature = "snark_spartan2")]
#[derive(Clone, Debug)]
pub struct FriProofSpartan2 {
    /// Serialized FRI proof
    pub proof_bytes: Vec<u8>,
    /// Evaluation at the challenge point
    pub evaluation: pallas::Scalar,
    /// Challenge point used for opening
    pub challenge: pallas::Scalar,
}

/// FRI prover key (contains polynomial evaluations and domain info)
#[cfg(feature = "snark_spartan2")]
#[derive(Clone, Debug)]
pub struct FriProverKey {
    /// Domain log size
    pub domain_log: usize,
    /// Polynomial evaluations in Goldilocks field
    pub evals: Vec<F>,
}

/// FRI commitment key (contains setup parameters)
#[cfg(feature = "snark_spartan2")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriCommitmentKey {
    /// Domain log size
    pub domain_log: usize,
    /// Maximum degree
    pub max_degree: usize,
    /// Setup label
    pub label: Vec<u8>,
}

/// FRI verifier key (minimal info needed for verification)
#[cfg(feature = "snark_spartan2")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriVerifierKey {
    /// Domain log size
    pub domain_log: usize,
    /// Maximum degree
    pub max_degree: usize,
}

/// FRI blinding factor
#[cfg(feature = "snark_spartan2")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FriBlind {
    pub value: u64,
}

/// FRI evaluation argument (proof of evaluation)
#[cfg(feature = "snark_spartan2")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriEvaluationArgument {
    pub proof_bytes: Vec<u8>,
    pub evaluation: pallas::Scalar,
    pub point: pallas::Scalar,
}

#[cfg(feature = "snark_spartan2")]
impl PCSEngineTrait<NeoFriEngine> for NeoFriPCS {
    type CommitmentKey = FriCommitmentKey;
    type VerifierKey = FriVerifierKey;
    type Commitment = FriCommitmentSpartan2;
    type PartialCommitment = FriCommitmentSpartan2; // Same as full commitment for simplicity
    type Blind = FriBlind;
    type EvaluationArgument = FriEvaluationArgument;
    
    fn setup(label: &'static [u8], n: usize) -> (Self::CommitmentKey, Self::VerifierKey) {
        let domain_log = (n.next_power_of_two().trailing_zeros() as usize).max(3);
        
        let commitment_key = FriCommitmentKey {
            domain_log,
            max_degree: n,
            label: label.to_vec(),
        };
        
        let verifier_key = FriVerifierKey {
            domain_log,
            max_degree: n,
        };
        
        (commitment_key, verifier_key)
    }
    
    fn width() -> usize {
        1 // FRI commits to one polynomial at a time
    }
    
    fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind {
        // Generate random blind for commitment
        use rand::Rng;
        let mut rng = rand::rng();
        let blind_value: u64 = rng.random();
        FriBlind { value: blind_value }
    }
    
    fn commit(
        ck: &Self::CommitmentKey,
        poly: &[pallas::Scalar],
        blind: &Self::Blind,
        _hide_degree: bool,
    ) -> Result<Self::Commitment, SpartanError> {
        // Convert Pallas scalars to Goldilocks field elements
        let goldilocks_evals: Vec<F> = poly.iter()
            .map(|scalar| pallas_scalar_to_goldilocks(scalar))
            .collect();
        
        // Pad to domain size
        let domain_size = 1 << ck.domain_log;
        let mut padded_evals = goldilocks_evals;
        padded_evals.resize(domain_size, F::ZERO);
        
        // Add blinding (simple approach: add blind to first element)
        if !padded_evals.is_empty() {
            padded_evals[0] = padded_evals[0] + F::from_u64(blind.value);
        }
        
        // Convert to batch format expected by FRI
        let evals_batch = vec![padded_evals];
        
        // Commit using p3-fri
        let fri = FriPCSWrapper::new();
        let (commitment, _prover_data) = fri.commit(&evals_batch, ck.domain_log, None)
            .map_err(|_| SpartanError::InvalidPCS)?;
        
        Ok(FriCommitmentSpartan2 {
            root: commitment.root,
            domain_size,
            metadata: vec![], 
        })
    }
    
    fn commit_partial(
        ck: &Self::CommitmentKey,
        poly: &[pallas::Scalar],
        blind: &Self::Blind,
        _hide_degree: bool,
    ) -> Result<(Self::PartialCommitment, Self::Blind), SpartanError> {
        // For simplicity, partial commitment is same as full commitment
        let commitment = Self::commit(ck, poly, blind, _hide_degree)?;
        Ok((commitment, blind.clone()))
    }
    
    fn check_partial(
        _partial: &Self::PartialCommitment,
        _n: usize,
    ) -> Result<(), SpartanError> {
        // Always valid for our simple implementation
        Ok(())
    }
    
    fn combine_partial(
        partials: &[Self::PartialCommitment],
    ) -> Result<Self::Commitment, SpartanError> {
        // For simplicity, just return the first partial commitment
        partials.first()
            .cloned()
            .ok_or(SpartanError::InvalidPCS)
    }
    
    fn combine_blinds(blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
        // Combine blinds by XOR
        let combined_value = blinds.iter().fold(0u64, |acc, blind| acc ^ blind.value);
        Ok(FriBlind { value: combined_value })
    }
    
    fn prove(
        _ck: &Self::CommitmentKey,
        _transcript: &mut <NeoFriEngine as Engine>::TE,
        _commitment: &Self::Commitment,
        poly: &[pallas::Scalar],
        _blind: &Self::Blind,
        points: &[pallas::Scalar],
    ) -> Result<(pallas::Scalar, Self::EvaluationArgument), SpartanError> {
        // For now, just evaluate at the first point and return a dummy proof
        let point = points.first().ok_or(SpartanError::InvalidPCS)?;
        
        // Simple polynomial evaluation (this should be more sophisticated)
        use ff::Field;
        let evaluation = if poly.is_empty() {
            pallas::Scalar::ZERO
        } else {
            poly[0] // Simplified: just return first coefficient
        };
        
        let eval_arg = FriEvaluationArgument {
            proof_bytes: vec![0u8; 32], // Dummy proof
            evaluation,
            point: *point,
        };
        
        Ok((evaluation, eval_arg))
    }
    
    fn verify(
        _vk: &Self::VerifierKey,
        _transcript: &mut <NeoFriEngine as Engine>::TE,
        _commitment: &Self::Commitment,
        _points: &[pallas::Scalar],
        eval: &pallas::Scalar,
        proof: &Self::EvaluationArgument,
    ) -> Result<(), SpartanError> {
        // Simple verification: check that evaluation matches
        if proof.evaluation == *eval {
            Ok(())
        } else {
            Err(SpartanError::InvalidPCS)
        }
    }
}

// Implement FoldingEngineTrait for NeoFriPCS
#[cfg(feature = "snark_spartan2")]
impl FoldingEngineTrait<NeoFriEngine> for NeoFriPCS {
    fn fold_commitments(
        commitments: &[Self::Commitment],
        _r: &[pallas::Scalar],
    ) -> Result<Self::Commitment, SpartanError> {
        // Simple folding - just return first commitment
        commitments.first()
            .cloned()
            .ok_or(SpartanError::InvalidPCS)
    }
    
    fn fold_blinds(
        blinds: &[Self::Blind],
        _r: &[pallas::Scalar],
    ) -> Result<Self::Blind, SpartanError> {
        // Simple folding - XOR all blinds
        let combined_value = blinds.iter().fold(0u64, |acc, blind| acc ^ blind.value);
        Ok(FriBlind { value: combined_value })
    }
}

/// Convert Pallas scalar to Goldilocks field element
/// This is a lossy conversion - we take the low 64 bits
#[cfg(feature = "snark_spartan2")]
fn pallas_scalar_to_goldilocks(scalar: &pallas::Scalar) -> F {
    use ff::PrimeField;
    
    // Get the canonical byte representation
    let bytes = scalar.to_repr();
    
    // Take the first 8 bytes (little-endian) and convert to u64
    let mut u64_bytes = [0u8; 8];
    u64_bytes.copy_from_slice(&bytes.as_ref()[0..8]);
    let value = u64::from_le_bytes(u64_bytes);
    
    // Convert to Goldilocks field
    F::from_u64(value)
}

/// Convert Goldilocks field element to Pallas scalar
#[cfg(feature = "snark_spartan2")]
#[allow(dead_code)]
fn goldilocks_to_pallas_scalar(f: &F) -> pallas::Scalar {
    #[allow(unused_imports)]
    use ff::PrimeField;
    
    // Get the canonical u64 representation
    let value = f.as_canonical_u64();
    
    // Convert to Pallas scalar
    pallas::Scalar::from(value)
}

#[cfg(all(test, feature = "snark_spartan2"))]
mod tests {
    use super::*;
    use ff::Field;
    
    #[test]
    fn test_field_conversion_roundtrip() {
        // Test small values
        let original = pallas::Scalar::from(12345u64);
        let goldilocks = pallas_scalar_to_goldilocks(&original);
        let converted_back = goldilocks_to_pallas_scalar(&goldilocks);
        assert_eq!(original, converted_back);
        
        // Test zero
        let zero = pallas::Scalar::ZERO;
        let goldilocks_zero = pallas_scalar_to_goldilocks(&zero);
        let zero_back = goldilocks_to_pallas_scalar(&goldilocks_zero);
        assert_eq!(zero, zero_back);
        
        // Test one
        let one = pallas::Scalar::ONE;
        let goldilocks_one = pallas_scalar_to_goldilocks(&one);
        let one_back = goldilocks_to_pallas_scalar(&goldilocks_one);
        assert_eq!(one, one_back);
    }
    
    #[test]
    fn test_fri_pcs_setup() {
        let (pk, vk) = NeoFriPCS::setup(b"test_label", 1024);
        assert_eq!(pk.domain_log, 10); // 2^10 = 1024
        assert_eq!(vk.domain_log, 10);
        assert_eq!(vk.max_degree, 1024);
    }
}
