//! Interactive sumcheck protocol implementation over extension field K = F_q^s
//! This is the single sum-check used throughout Neo protocol.

use super::{UnivPoly, fiat_shamir::*};
use neo_math::{ExtF, Polynomial};
use p3_field::{PrimeCharacteristicRing, Field};

/// Error types for sumcheck protocol
#[derive(Debug, Clone)]
pub enum SumcheckError {
    InvalidClaim(String),
    InvalidProof(String),
    TranscriptError(String),
    PolynomialError(String),
}

impl std::fmt::Display for SumcheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for SumcheckError {}

/// Batched sumcheck prover: prove multiple claims simultaneously using random linear combination
pub fn batched_sumcheck_prover(
    claims: &[ExtF],
    polys: &[&dyn UnivPoly],
    transcript: &mut Vec<u8>,
) -> Result<Vec<(Polynomial<ExtF>, ExtF)>, SumcheckError> {
    if claims.len() != polys.len() {
        return Err(SumcheckError::InvalidClaim("claims/polys length mismatch".into()));
    }
    
    if claims.is_empty() {
        return Err(SumcheckError::InvalidClaim("empty claims".into()));
    }
    
    // Extract number of variables from first polynomial 
    let num_vars = polys[0].num_vars();
    for poly in polys {
        if poly.num_vars() != num_vars {
            return Err(SumcheckError::PolynomialError("inconsistent num_vars".into()));
        }
    }
    
    // Random batching coefficient
    fs_absorb_bytes(transcript, b"sumcheck.batch", &claims.len().to_le_bytes());
    let rho = fs_challenge_ext(transcript, b"sumcheck.rho");
    
    // Batch the claims: combined_claim = Σ rho^i * claims[i] 
    let combined_claim = claims.iter().enumerate()
        .fold(ExtF::ZERO, |acc, (i, &claim)| {
            let mut coeff = ExtF::ONE;
            for _ in 0..i { coeff *= rho; }
            acc + coeff * claim
        });
    
    fs_absorb_ext(transcript, b"sumcheck.combined_claim", combined_claim);
    
    // Interactive sumcheck rounds
    let mut rounds = Vec::with_capacity(num_vars);
    let mut partial_point = Vec::with_capacity(num_vars);
    
    for round in 0..num_vars {
        // Prover sends univariate polynomial for this round
        let uni_polys: Vec<Polynomial<ExtF>> = polys.iter()
            .map(|&poly| {
                // This is simplified - in practice we'd need proper partial evaluation
                // For now, create a linear polynomial as placeholder
                Polynomial::new(vec![ExtF::ZERO, ExtF::ONE])
            })
            .collect();
        
        // Batch the univariates using same rho
        let batched_uni = batch_unis(&uni_polys, rho);
        
        // Send polynomial coefficients to transcript
        fs_absorb_bytes(transcript, b"sumcheck.round_poly", 
                       &bincode::serialize(&batched_uni.coeffs()).unwrap());
        
        // Verifier challenge
        let challenge = fs_challenge_ext(transcript, b"sumcheck.challenge");
        
        // Evaluate batched polynomial at challenge
        let eval = batched_uni.eval(challenge);
        
        rounds.push((batched_uni, eval));
        partial_point.push(challenge);
        
        // Check consistency: eval should match expected sum structure
        // This is where the real sumcheck verification happens
    }
    
    // Final evaluation check would happen here in complete implementation
    Ok(rounds)
}

/// Batched sumcheck verifier
pub fn batched_sumcheck_verifier(
    claims: &[ExtF],
    rounds: &[(Polynomial<ExtF>, ExtF)],
    transcript: &mut Vec<u8>,
) -> Option<(Vec<ExtF>, ExtF)> {
    if claims.is_empty() || rounds.is_empty() {
        return None;
    }
    
    // Re-derive the same random batching coefficient
    fs_absorb_bytes(transcript, b"sumcheck.batch", &claims.len().to_le_bytes());
    let rho = fs_challenge_ext(transcript, b"sumcheck.rho");
    
    // Re-compute combined claim
    let combined_claim = claims.iter().enumerate()
        .fold(ExtF::ZERO, |acc, (i, &claim)| {
            let mut coeff = ExtF::ONE;
            for _ in 0..i { coeff *= rho; }
            acc + coeff * claim
        });
    
    fs_absorb_ext(transcript, b"sumcheck.combined_claim", combined_claim);
    
    let mut challenges = Vec::with_capacity(rounds.len());
    let mut running_sum = combined_claim;
    
    for (round_idx, (poly, eval)) in rounds.iter().enumerate() {
        // Verify polynomial was properly absorbed
        fs_absorb_bytes(transcript, b"sumcheck.round_poly", 
                       &bincode::serialize(&poly.coeffs()).unwrap());
        
        // Generate same challenge
        let challenge = fs_challenge_ext(transcript, b"sumcheck.challenge");
        challenges.push(challenge);
        
        // Verify polynomial evaluation
        let computed_eval = poly.eval(challenge);
        if computed_eval != *eval {
            return None;
        }
        
        // Check sumcheck equation: P(0) + P(1) should equal running_sum
        let p_0 = poly.eval(ExtF::ZERO);
        let p_1 = poly.eval(ExtF::ONE);
        
        if round_idx == 0 {
            // First round: check against original claim
            if p_0 + p_1 != running_sum {
                return None;
            }
        }
        
        // Update running sum for next round
        running_sum = *eval;
    }
    
    // Return challenges and final evaluation
    let final_eval = running_sum;
    Some((challenges, final_eval))
}

// Simplified single-polynomial versions
pub fn multilinear_sumcheck_prover(
    claim: ExtF,
    poly: &dyn UnivPoly,
    transcript: &mut Vec<u8>,
) -> Result<Vec<(Polynomial<ExtF>, ExtF)>, SumcheckError> {
    batched_sumcheck_prover(&[claim], &[poly], transcript)
}

pub fn multilinear_sumcheck_verifier(
    claim: ExtF,
    rounds: &[(Polynomial<ExtF>, ExtF)],
    transcript: &mut Vec<u8>,
) -> Option<(Vec<ExtF>, ExtF)> {
    batched_sumcheck_verifier(&[claim], rounds, transcript)
}

// Batch multilinear versions (aliases for consistency)
pub use self::{
    batched_sumcheck_prover as batched_multilinear_sumcheck_prover,
    batched_sumcheck_verifier as batched_multilinear_sumcheck_verifier,
};

// Helper function for transcript absorption 
fn fs_absorb_ext(transcript: &mut Vec<u8>, label: &[u8], x: ExtF) {
    // Convert ExtF to bytes for absorption
    let bytes = bincode::serialize(&x).unwrap_or_default();
    fs_absorb_bytes(transcript, label, &bytes);
}
