pub mod fiat_shamir;
pub mod challenger;

pub use challenger::NeoChallenger;
pub mod poly;
pub mod sumcheck;

pub use fiat_shamir::{
    batch_unis, fiat_shamir_challenge, fiat_shamir_challenge_base,
    fs_absorb_bytes, fs_challenge_ext, fs_challenge_base_labeled, 
    fs_challenge_ext_labeled, fs_challenge_u64_labeled, Transcript
};
pub use poly::{MultilinearEvals, UnivPoly};
pub use sumcheck::{
    batched_multilinear_sumcheck_prover, batched_multilinear_sumcheck_verifier,
    batched_sumcheck_prover, batched_sumcheck_verifier, multilinear_sumcheck_prover,
    multilinear_sumcheck_verifier,
};

pub use neo_fields::{from_base, ExtF, ExtFieldNormTrait, F};
pub use neo_poly::Polynomial;
pub use neo_modint::{Coeff, ModInt};

// Spartan2 integration
pub mod spartan2_sumcheck {
    use super::*;
    use neo_fields::spartan2_engine::GoldilocksEngine;
    use spartan2::sumcheck::SumcheckProof;
    use spartan2::traits::Engine;
    
    /// Spartan2-based batched sum-check prover
    pub fn spartan2_batched_sumcheck_prover<P>(
        _polys: &[P],
        _transcript: &mut <GoldilocksEngine as Engine>::TE,
    ) -> Result<SumcheckProof<GoldilocksEngine>, String>
    where
        P: Fn(&[F]) -> F + Send + Sync,
    {
        // Convert our polynomial closures to Spartan2 format
        // This is a simplified version - full implementation would handle
        // the multilinear polynomial evaluation properly
        
        // For now, return a placeholder - this would need proper implementation
        // based on the specific Spartan2 sumcheck API
        todo!("Implement Spartan2 sumcheck integration")
    }
    
    /// Spartan2-based batched sum-check verifier  
    pub fn spartan2_batched_sumcheck_verifier(
        _proof: &SumcheckProof<GoldilocksEngine>,
        _transcript: &mut <GoldilocksEngine as Engine>::TE,
        _expected_sum: F,
        _num_vars: usize,
    ) -> Result<Vec<F>, String> {
        // Verify the Spartan2 sumcheck proof
        // This would delegate to Spartan2's verification logic
        todo!("Implement Spartan2 sumcheck verification")
    }
}

pub use spartan2_sumcheck::{spartan2_batched_sumcheck_prover, spartan2_batched_sumcheck_verifier};
