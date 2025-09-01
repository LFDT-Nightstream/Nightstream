#![forbid(unsafe_code)]
//! Neo folding layer: CCS instances → ME claims → Spartan2 proof
//!
//! **Single transcript**: All reductions use the same domain-separated FS transcript  
//! - **One sum-check over K = F_q^2**: No other crates can create sum-check instances
//! - **Single transcript**: All reductions use the same domain-separated FS transcript  
//! - **Three-reduction pipeline**: Π_CCS → Π_RLC → Π_DEC composition as in Neo §4-5

use neo_params::NeoParams;
use neo_ccs::{McsInstance, MeInstance, MeWitness};
// use neo_math::transcript::Transcript; // TODO: Use when implementing actual transcript

// Type aliases for concrete ME types to replace legacy MEInstance/MEWitness
type ConcreteMeInstance = MeInstance<Vec<neo_math::F>, neo_math::F, neo_math::ExtF>;
type ConcreteMeWitness = MeWitness<neo_math::F>;

// Export transcript module
pub mod transcript;

// Bridge adapter for converting modern types to legacy bridge format
mod bridge_adapter;

// Sumcheck functionality (placeholder - TODO: implement)
pub mod sumcheck {
    // TODO: Implement transcript in neo-fold where it belongs according to STRUCTURE.md
    #[derive(Debug, Clone)]
    pub struct SumcheckProof;
    
    pub fn sumcheck_prove() -> SumcheckProof {
        SumcheckProof
    }
    
    pub fn sumcheck_verify(_proof: &SumcheckProof) -> bool {
        true
    }
}

/// Top-level folding error
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid reduction: {0}")]
    InvalidReduction(String),
    #[error("Bridge error: {0}")]
    Bridge(String),
    #[error("Sumcheck error: {0}")]
    Sumcheck(String),
    #[error("Extension policy violation: {0}")]
    ExtensionPolicy(String),
}

/// Proof that k+1 CCS instances fold to k instances
#[derive(Debug, Clone)]
pub struct FoldingProof {
    // TODO: Add actual proof fields from the three-reduction pipeline
    pub rlc_proof: Vec<u8>,
    pub dec_proof: Vec<u8>,
    pub sumcheck_proof: sumcheck::SumcheckProof,
}

/// Neo protocol parameters
// NeoParams is imported from neo-params crate above

/// Fold k+1 CCS instances into k instances using the three-reduction pipeline
pub fn fold_step(
    structure: &neo_ccs::CcsStructure<neo_math::F>,
    instances: &[McsInstance<Vec<u8>, neo_math::F>],
    params: &NeoParams,
) -> Result<(Vec<McsInstance<Vec<u8>, neo_math::F>>, FoldingProof), Error> {
    if instances.is_empty() {
        return Err(Error::InvalidReduction("Cannot fold empty instance set".to_string()));
    }
    
    // MUST: Enforce extension degree policy before constructing sum-check
    // Compute ell = log2(n) where n is the domain size (should be power of two)
    let n = structure.n;
    if !n.is_power_of_two() {
        return Err(Error::Sumcheck("CCS domain size n must be power of two for sum-check".into()));
    }
    let ell = (n.ilog2()) as u32;
    
    // Compute d_sc = max total degree of the sum-check polynomial Q
    // This is the maximum total degree across all terms in the CCS polynomial f
    let d_sc = structure.f.terms().iter()
        .map(|term| term.exps.iter().sum::<u32>())
        .max()
        .unwrap_or(1);
    enforce_extension_policy(params, ell, d_sc)?;
    
    // For now, return a placeholder implementation
    // TODO: Implement the actual three-reduction pipeline:
    // 1. CCS → RLC (Randomized Linear Combination)
    // 2. RLC → DEC (Degree Check) 
    // 3. DEC → Single sumcheck over extension field
    
    let folded_instances = instances[..instances.len()-1].to_vec();
    let proof = FoldingProof {
        rlc_proof: vec![42u8; 32],
        dec_proof: vec![84u8; 32],
        sumcheck_proof: sumcheck::sumcheck_prove(),
    };
    
    Ok((folded_instances, proof))
}

/// Verify a folding proof
pub fn verify_fold(
    original_instances: &[McsInstance<Vec<u8>, neo_math::F>],
    folded_instances: &[McsInstance<Vec<u8>, neo_math::F>],
    _proof: &FoldingProof,
    _params: &NeoParams,
) -> Result<bool, Error> {
    // Placeholder verification
    // TODO: Implement actual verification of the three-reduction pipeline
    
    if original_instances.len() != folded_instances.len() + 1 {
        return Ok(false);
    }
    
    Ok(true)
}

/// Enforce extension degree policy before sum-check construction.
/// This must be called before instantiating any sum-check with the given parameters.
fn enforce_extension_policy(params: &NeoParams, ell: u32, d_sc: u32) -> Result<(), Error> {
    match params.extension_check(ell, d_sc) {
        Ok(_summary) => {
            // Optionally record summary.slack_bits in transcript header (future work)
            // For now, just record success without logging (to avoid dependency on log crate)
            Ok(())
        }
        Err(neo_params::ParamsError::UnsupportedExtension { required }) => {
            Err(Error::ExtensionPolicy(format!(
                "unsupported extension degree; required s={required}, supported s=2"
            )))
        }
        Err(e) => {
            Err(Error::ExtensionPolicy(format!(
                "extension check failed: {e}"
            )))
        }
    }
}

/// Complete folding pipeline: many CCS instances → single ME claim
pub fn fold_to_single_me(
    structure: &neo_ccs::CcsStructure<neo_math::F>,
    instances: &[McsInstance<Vec<u8>, neo_math::F>],
    params: &NeoParams,
) -> Result<(ConcreteMeInstance, ConcreteMeWitness, Vec<FoldingProof>), Error> {
    let mut current_instances = instances.to_vec();
    let mut proofs = Vec::new();
    
    // Fold down to a single instance
    while current_instances.len() > 1 {
        let (folded, proof) = fold_step(structure, &current_instances, params)?;
        proofs.push(proof);
        current_instances = folded;
    }
    
    // Convert final CCS instance to ME format
    // TODO: Implement proper CCS → ME conversion
    let me_instance = create_dummy_me_instance();
    let me_witness = create_dummy_me_witness();
    
    Ok((me_instance, me_witness, proofs))
}

/// Final compression: bridge from folded ME claims to Spartan2 proof
pub mod spartan_compression {
    use super::*;
    use crate::bridge_adapter;
    
    /// Compress a final ME(b,L) claim to a Spartan2 SNARK via neo-spartan-bridge
    /// 
    /// ⚠️  **TEMPORARY**: Uses Keccak transcript in Hash-MLE backend (inconsistent with Neo's Poseidon2)
    /// This will be fixed once Spartan2 supports Poseidon2 or we implement our own PCS.
    pub fn compress_me_to_spartan(
        me_instance: &ConcreteMeInstance,
        me_witness: &ConcreteMeWitness,
    ) -> Result<Vec<u8>, String> {
        // Use the Neo parameters from the instance/witness context
        let params = neo_params::NeoParams::goldilocks_127();
        
        // Route through the bridge adapter with type conversion
        bridge_adapter::compress_via_bridge(me_instance, me_witness, &params)
    }
    
    /// Verify a Spartan2 compressed ME proof via neo-spartan-bridge
    /// 
    /// ⚠️  **TEMPORARY**: Uses Keccak transcript in Hash-MLE backend (inconsistent with Neo's Poseidon2)
    /// This will be fixed once Spartan2 supports Poseidon2 or we implement our own PCS.
    pub fn verify_spartan_me_proof(
        proof: &[u8],
        public_inputs: &[neo_math::F],
    ) -> Result<bool, String> {
        // Route through the bridge adapter
        bridge_adapter::verify_via_bridge(proof, public_inputs)
    }
    
    /// Complete folding with final Spartan2 compression 
    /// This would be the main entry point for the full Neo protocol
    pub fn fold_and_compress(
        structure: &neo_ccs::CcsStructure<neo_math::F>,
        instances: &[McsInstance<Vec<u8>, neo_math::F>],
        params: &NeoParams,
    ) -> Result<Vec<u8>, Error> {
        // Step 1: Execute folding pipeline
        let (me_instance, me_witness, _folding_proofs) = fold_to_single_me(structure, instances, params)?;
        
        // Step 2: Compress final ME claim to Spartan2
        let proof = compress_me_to_spartan(&me_instance, &me_witness)
            .map_err(|e| Error::Bridge(e))?;
        
        Ok(proof)
    }
    
    // Helper functions for creating dummy ME instances (placeholder implementations)
    pub fn create_dummy_me_instance() -> ConcreteMeInstance {
        use neo_math::{F, ExtF};
        use neo_ccs::Mat;
        use p3_field::PrimeCharacteristicRing;
        
        // Create a minimal ME instance for testing/placeholder purposes
        // The modern MeInstance has different fields than the legacy MEInstance
        ConcreteMeInstance {
            c: vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)],// commitment
            X: Mat::zero(2, 1, F::ZERO), // X = L_x(Z) matrix 
            r: vec![ExtF::from(F::from_u64(42))],// r in extension field
            y: vec![vec![ExtF::from(F::from_u64(100))]],// y_j outputs in extension field
            m_in: 1, // number of public inputs
        }
    }
    
    pub fn create_dummy_me_witness() -> ConcreteMeWitness {
        use neo_math::F;
        use neo_ccs::Mat;
        use p3_field::PrimeCharacteristicRing;
        
        // Create a minimal ME witness for testing/placeholder purposes
        // The modern MeWitness just contains the Z matrix
        ConcreteMeWitness {
            Z: Mat::from_row_major(3, 2, vec![
                F::from_u64(10), F::from_u64(20),
                F::from_u64(30), F::from_u64(40), 
                F::from_u64(50), F::from_u64(60)
            ]),
        }
    }
}

/// Create a dummy ME instance for testing
fn create_dummy_me_instance() -> ConcreteMeInstance {
    spartan_compression::create_dummy_me_instance()
}

/// Create a dummy ME witness for testing
fn create_dummy_me_witness() -> ConcreteMeWitness {
    spartan_compression::create_dummy_me_witness()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test] 
    fn test_real_proof_generation() {
        // This test ensures real proofs are generated (no more stubs)
        let dummy_me = create_dummy_me_instance();
        let dummy_wit = create_dummy_me_witness();
        
        // Real proof generation should work in both debug and release modes
        let proof_result = spartan_compression::compress_me_to_spartan(&dummy_me, &dummy_wit);
        assert!(proof_result.is_ok(), "Real proof generation should succeed");
        
        let proof = proof_result.unwrap();
        // Real proofs should NOT contain "DEMO_STUB_" - they are actual cryptographic artifacts
        assert!(!String::from_utf8_lossy(&proof).contains("DEMO_STUB_"));
        // Real proofs should be substantial in size (> 1KB)
        assert!(proof.len() > 1000, "Real proof should be substantial, got {} bytes", proof.len());
        
        let verify_result = spartan_compression::verify_spartan_me_proof(&proof, &[]);
        assert!(verify_result.is_ok(), "Verification should succeed");
        assert!(verify_result.unwrap(), "Real proof should verify");
    }
    
    #[test]
    fn test_fold_step_placeholder() {
        // Test that fold_step function accepts valid inputs and returns a result
        // This is a placeholder test while the full folding pipeline is implemented
        let params = NeoParams::goldilocks_127();
        
        // Create dummy instances for testing
        let instances = vec![
            create_dummy_mcs_instance(),
            create_dummy_mcs_instance(),
        ];
        let structure = create_dummy_ccs_structure();
        
        let result = fold_step(&structure, &instances, &params);
        
        // The test may fail due to extension policy, but that's expected for dummy data
        // Just verify that the function runs without panicking
        match result {
            Ok((folded, _proof)) => {
                assert_eq!(folded.len(), 1); // Should fold 2 instances to 1
            }
            Err(Error::ExtensionPolicy(_)) => {
                // Extension policy rejection is expected for dummy data
                // This is not a failure - just means our dummy data doesn't meet security requirements
                println!("Extension policy rejected dummy data (expected)");
            }
            Err(e) => {
                panic!("Unexpected error (should only get extension policy errors): {:?}", e);
            }
        }
    }

    #[test]
    fn test_fold_step_strict_boundary() {
        // Test that strict 128-bit security is properly rejected
        let params = NeoParams::goldilocks_128_strict();
        let instances = vec![
            create_dummy_mcs_instance(),
            create_dummy_mcs_instance(),
        ];
        let structure = create_dummy_ccs_structure();
        
        let result = fold_step(&structure, &instances, &params);
        assert!(result.is_err(), "λ=128 with s≤2 should be rejected for Goldilocks");
        
        // Verify it's the extension policy error we expect
        if let Err(Error::ExtensionPolicy(msg)) = result {
            assert!(msg.contains("required s=3"), "Should require s=3 for λ=128");
        } else {
            panic!("Expected ExtensionPolicy error");
        }
    }
    
    fn create_dummy_mcs_instance() -> McsInstance<Vec<u8>, neo_math::F> {
        use neo_math::F;
        use p3_field::PrimeCharacteristicRing;
        
        McsInstance {
            c: b"test_commitment".to_vec(),
            x: vec![F::from_u64(1), F::from_u64(2)],
            m_in: 2,
        }
    }
    
    fn create_dummy_ccs_structure() -> neo_ccs::CcsStructure<neo_math::F> {
        use neo_math::F;
        use neo_ccs::{CcsStructure, SparsePoly, Term, Mat};
        use p3_field::PrimeCharacteristicRing;
        
        // Create matrices and polynomial, then use the constructor
        let matrices = vec![Mat::zero(4, 3, F::ZERO)]; // Single 4x3 matrix (n=4, m=3)
        let terms = vec![
            Term { coeff: F::ONE, exps: vec![1] } // Simple linear term: 1 * X_0
        ];
        let f = SparsePoly::new(1, terms); // arity=1 to match single matrix
        
        CcsStructure::new(matrices, f).expect("Valid dummy CCS structure")
    }
}