pub mod fiat_shamir;
pub mod challenger;

// Re-export the unified transcript for backward compatibility
pub use crate::transcript::FoldTranscript as Transcript;

#[deprecated(since = "0.1.0", note = "Use crate::transcript::FoldTranscript instead")]
pub use challenger::NeoChallenger;
pub mod poly;
pub mod sumcheck;

pub use fiat_shamir::{
    batch_unis, fiat_shamir_challenge, fiat_shamir_challenge_base,
    fs_absorb_bytes, fs_challenge_ext, fs_challenge_base_labeled, 
    fs_challenge_ext_labeled, fs_challenge_u64_labeled
};
pub use poly::{MultilinearEvals, UnivPoly};
pub use sumcheck::{
    batched_multilinear_sumcheck_prover, batched_multilinear_sumcheck_verifier,
    batched_sumcheck_prover, batched_sumcheck_verifier, multilinear_sumcheck_prover,
    multilinear_sumcheck_verifier,
};

pub use neo_math::{from_base, ExtF, F, Polynomial, Coeff, ModInt, RingElement};

// Spartan2 integration
// Spartan2 integration (shim). This compiles now and keeps public API stable.
// Provides SNARK-compatible sumcheck protocols with Spartan2 integration.

pub mod spartan2_sumcheck {
    use super::*;
    
    /// Backend-agnostic proof for "Spartan2 mode".
    /// Today this is just your NARK messages. Later you can add a real Spartan2 variant.
    #[derive(Clone, Debug)]
    pub struct Spartan2SumcheckProof {
        pub rounds: Vec<(Polynomial<ExtF>, ExtF)>,
    }

    /// Prover shim: use your NARK batched sum-check and wrap its messages.
    pub fn spartan2_batched_sumcheck_prover(
        claims: &[ExtF],
        polys: &[&dyn UnivPoly],
        transcript: &mut Vec<u8>,
    ) -> Result<Spartan2SumcheckProof, String> {
        let rounds = crate::sumcheck::batched_sumcheck_prover(claims, polys, transcript)
            .map_err(|e| format!("sumcheck prover failed: {e}"))?;
        Ok(Spartan2SumcheckProof { rounds })
    }

    /// Verifier shim: delegate to your NARK verifier.
    /// Returns the verifier's challenges `r` if successful.
    pub fn spartan2_batched_sumcheck_verifier(
        claims: &[ExtF],
        proof: &Spartan2SumcheckProof,
        transcript: &mut Vec<u8>,
    ) -> Result<Vec<ExtF>, String> {
        let (challenges, _final_eval) = crate::sumcheck::batched_sumcheck_verifier(claims, &proof.rounds, transcript)
            .ok_or_else(|| "sumcheck verification failed".to_string())?;
        Ok(challenges)
    }
}

// Re-export (same names as before, but now they actually work)
pub use spartan2_sumcheck::{
    spartan2_batched_sumcheck_prover, spartan2_batched_sumcheck_verifier, Spartan2SumcheckProof,
};

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use crate::{MultilinearEvals, UnivPoly, from_base, F, ExtF};
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn spartan2_sumcheck_shim_round_trip() {
        // f(x,y) = 3x + 5y - 2xy over {0,1}^2
        let evals = vec![
            from_base(F::from_u64(0)), // (0,0)
            from_base(F::from_u64(5)), // (0,1)
            from_base(F::from_u64(3)), // (1,0)
            from_base(F::from_u64(6)), // (1,1): 3 + 5 - 2
        ];
        let mle = MultilinearEvals::new(evals.clone());
        let sum_claim = evals.iter().copied().fold(ExtF::ZERO, |a,b| a+b);

        let mut transcript = Vec::new();
        let proof = super::spartan2_sumcheck::spartan2_batched_sumcheck_prover(
            &[sum_claim],
            &[&mle as &dyn UnivPoly],
            &mut transcript
        ).expect("prover");

        // Verifier starts with fresh transcript (same initial state as prover)
        let mut transcript_v = Vec::new();
        let _chals = super::spartan2_sumcheck::spartan2_batched_sumcheck_verifier(
            &[sum_claim],
            &proof,
            &mut transcript_v
        ).expect("verify");
        
        println!("✅ Spartan2 sumcheck shim round-trip test passed");
    }
}
