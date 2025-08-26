use std::time::Instant;
use thiserror::Error;

use neo_ccs::{CcsInstance, CcsStructure, CcsWitness, check_satisfiability};
use neo_commit::AjtaiCommitter;
use neo_fold::Proof;
#[allow(unused_imports)]
use neo_fold::FoldState;

/// Orchestrator errors (kept minimal on purpose)
#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("ccs constraints not satisfied by provided witness")]
    Unsatisfied,
}

/// Minimal timing/size metrics returned alongside the proof.
#[derive(Clone, Debug)]
pub struct Metrics {
    pub prove_ms: f64,
    pub proof_bytes: usize,
}

/// PROVE: run the SNARK pipeline over a CCS + (instance, witness).
///
/// - Accepts a *prepared* CCS instance (you already committed in main).
/// - Auto-detects the number of sum-check rounds from the CCS (handled inside neo-fold).
/// - Uses Spartan2 for succinct SNARK proofs.
pub fn prove(
    ccs: &CcsStructure,
    instance: &CcsInstance,
    witness: &CcsWitness,
) -> Result<(Proof, Metrics), OrchestratorError> {
    if !check_satisfiability(ccs, instance, witness) {
        return Err(OrchestratorError::Unsatisfied);
    }

    // Enforce secure parameters in production
    let _committer = AjtaiCommitter::new(); // This enforces secure params

    let t0 = Instant::now();
    
    // Always use Spartan2 SNARK mode
    use neo_fold::spartan_ivc::{spartan_compress, domain_separated_transcript};
    
    let transcript = domain_separated_transcript(0, "neo_orchestrator_prove");
    let (proof_bytes, vk_bytes) = spartan_compress(ccs, instance, witness, &transcript)
        .map_err(|e| {
            eprintln!("Spartan2 SNARK proof generation failed: {}", e);
            OrchestratorError::Unsatisfied
        })?;
    
    // Create proof with both proof and VK embedded
    let mut combined_transcript = proof_bytes;
    combined_transcript.extend_from_slice(b"||VK||");
    combined_transcript.extend_from_slice(&vk_bytes);
    let proof = Proof { transcript: combined_transcript };
    
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let proof_bytes = proof.transcript.len();
    Ok((proof, Metrics { prove_ms, proof_bytes }))
}

/// VERIFY: check a SNARK transcript against the CCS.
///
/// Returns `true` on success. Always expects Spartan2 SNARK proofs.
pub fn verify(ccs: &CcsStructure, proof: &Proof) -> bool {
    let _committer = AjtaiCommitter::new(); // Uses secure params
    
    // Parse Spartan2 SNARK proof (must contain VK separator)
    if let Some(vk_pos) = proof.transcript.windows(6).position(|w| w == b"||VK||") {
        let proof_bytes = &proof.transcript[..vk_pos];
        let vk_bytes = &proof.transcript[vk_pos + 6..];
        
        // Create a dummy instance for verification (in a real implementation,
        // the instance would be embedded in the proof or provided separately)
        use neo_fields::F;
        use p3_field::PrimeCharacteristicRing;
        let dummy_instance = neo_ccs::CcsInstance { 
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ZERO,
        };
        
        use neo_fold::spartan_ivc::{spartan_verify, domain_separated_transcript};
        let transcript = domain_separated_transcript(0, "neo_orchestrator_verify");
        
        match spartan_verify(proof_bytes, vk_bytes, ccs, &dummy_instance, &transcript) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Spartan2 SNARK verification failed: {}", e);
                false
            }
        }
    } else {
        eprintln!("Invalid proof format: expected Spartan2 SNARK proof with VK separator");
        false // Malformed or non-SNARK proof
    }
}
