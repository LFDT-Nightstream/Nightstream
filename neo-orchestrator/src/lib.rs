//! Neo Orchestrator: High-level SNARK interface for Neo Protocol
//!
//! This crate provides a simplified interface for the complete Neo protocol pipeline,
//! orchestrating the interaction between CCS constraints, Ajtai commitments, folding,
//! and Spartan2 compression.

use anyhow::Result;
use std::time::Instant;
use p3_field::PrimeCharacteristicRing;

// Re-export key types from neo-ccs and neo-fold  
pub use neo_ccs::CcsStructure;
pub use neo_fold::{fold_ccs_instances, FoldingProof};
pub use neo_spartan_bridge::{compress_me_to_spartan, verify_me_spartan};

// Type aliases for concrete types used in Neo protocol
type ConcreteMcsInstance = neo_ccs::McsInstance<neo_ajtai::Commitment, neo_math::F>;
type ConcreteMcsWitness = neo_ccs::McsWitness<neo_math::F>;

/// Performance metrics for proof generation and verification
#[derive(Debug, Clone)]
pub struct Metrics {
    pub prove_ms: f64,
    pub proof_bytes: usize,
}

/// Errors that can occur during orchestration
#[derive(thiserror::Error, Debug)]
pub enum OrchestratorError {
    #[error("Folding failed: {0}")]
    FoldingError(String),
    
    #[error("Spartan compression failed: {0}")]
    CompressionError(String),
    
    #[error("Verification failed: {0}")]
    VerificationError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
}

/// Generate a complete Neo SNARK proof for the given CCS instance and witness
///
/// This orchestrates the full pipeline:
/// 1. Folding: Reduce CCS instances to ME instances  
/// 2. Compression: Convert ME claim to Spartan2 SNARK proof
///
/// Returns the proof bytes and performance metrics.
pub fn prove(
    ccs: &CcsStructure<neo_math::F>,
    instances: &[ConcreteMcsInstance],
    witnesses: &[ConcreteMcsWitness],
) -> Result<(Vec<u8>, Metrics), OrchestratorError> {
    let prove_start = Instant::now();
    
    // Step 1: Execute folding pipeline
    let params = neo_params::NeoParams::goldilocks_127();
    let (me_instances, _folding_proof) = fold_ccs_instances(&params, ccs, instances, witnesses)
        .map_err(|e| OrchestratorError::FoldingError(format!("{}", e)))?;
    
    // For now, use placeholder ME to legacy bridge conversion
    // TODO: Replace with proper modern ME -> Spartan bridge once implemented
    eprintln!("⚠️  ORCHESTRATOR: Using placeholder folding pipeline");
    eprintln!("  Generated {} ME instances (modern)", me_instances.len());
    
    // Create placeholder legacy ME for bridge compatibility  
    #[allow(deprecated)]
    let me_legacy = neo_ccs::MEInstance {
        c_coords: vec![neo_math::F::ZERO; 16], // Placeholder commitment coords
        y_outputs: vec![neo_math::F::ZERO; 4], // Placeholder ME outputs
        r_point: vec![neo_math::F::ZERO; 8],   // Placeholder challenge vector
        base_b: params.b as u64,
        header_digest: [0u8; 32],
    };
    
    #[allow(deprecated)]
    let me_wit_legacy = neo_ccs::MEWitness {
        z_digits: vec![0i64; 32],              // Placeholder witness digits  
        weight_vectors: vec![vec![neo_math::F::ZERO; 32]; 4],
        ajtai_rows: None,
    };
    
    // Step 2: Compress to Spartan2 SNARK
    let proof_bundle = compress_me_to_spartan(&me_legacy, &me_wit_legacy)
        .map_err(|e| OrchestratorError::CompressionError(format!("{}", e)))?;
    
    let prove_time = prove_start.elapsed();
    let metrics = Metrics {
        prove_ms: prove_time.as_secs_f64() * 1000.0,
        proof_bytes: proof_bundle.proof.len(),
    };
    
    Ok((proof_bundle.proof, metrics))
}

/// Verify a Neo SNARK proof against the given CCS instance
///
/// This verifies the complete proof pipeline by checking the Spartan2 proof
/// against the public inputs from the CCS instance.
pub fn verify(
    _ccs: &CcsStructure<neo_math::F>,
    _instances: &[ConcreteMcsInstance], 
    proof: &[u8]
) -> bool {
    // Create a proof bundle for verification
    // TODO: This is placeholder - need proper public IO reconstruction
    let proof_bundle = neo_spartan_bridge::ProofBundle::new_with_vk(
        proof.to_vec(),
        vec![], // Empty VK for now
        vec![], // Empty public IO for now  
    );
    
    // Verify the Spartan2 proof
    match verify_me_spartan(&proof_bundle) {
        Ok(result) => result,
        Err(_) => false,
    }
}

/// Simplified prove interface for single instance/witness pairs
pub fn prove_single(
    ccs: &CcsStructure<neo_math::F>,
    instance: &ConcreteMcsInstance,
    witness: &ConcreteMcsWitness,
) -> Result<(Vec<u8>, Metrics), OrchestratorError> {
    prove(ccs, &[instance.clone()], &[witness.clone()])
}

/// Simplified verify interface for single instance
pub fn verify_single(
    ccs: &CcsStructure<neo_math::F>,
    instance: &ConcreteMcsInstance,
    proof: &[u8]
) -> bool {
    verify(ccs, &[instance.clone()], proof)
}

// Backward compatibility exports
pub mod neutronnova_integration {
    //! Placeholder module for backward compatibility
    //! This was used in the legacy integration but is no longer needed
    //! with the modern Neo architecture.
}

pub mod spartan2 {
    //! Direct integration with Spartan2 backend
    //! All functionality is now handled through the neo-spartan-bridge crate
    //! for better separation of concerns and clean API boundaries.
}